"""Character LoRA for Wan2.2 TI2V-5B.

Freezes the DiT, the T5 encoder, and the VAE. Trains a rank-16 LoRA with
flow-matching loss. Windows are taken at even starts inside each clip.

With --clip-ref, --ref-prob of the steps pin a different frame from the same
clip into latent frame 0 at timestep 0 and leave that frame out of the loss.
The remaining frames are the sampled window, so the loss asks for that
window's motion rather than the pinned pose. The other steps noise every
frame and score all of them. The reference sheets are not used.

With --pin-first, every step keeps the first latent frame clean at timestep
0 and leaves it out of the loss. That frame is the encoded window, so the
later frames continue the clip's own pose. --sheet-prob replaces it with the
front, side, or back reference on that fraction of steps. --attn-only skips
the feed-forward layers.

Without either flag, --anchor-prob is the chance of pinning the reference
sheet. The other steps noise every frame and score all of them.

Does not load the anime style LoRA. Saved weights use the same
diffusion_model.*.lora_A/B layout that scripts/run_lora.py already applies.
Apply this file on its own, at scale 1.0.
"""

import argparse
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

WAN_REPO = Path("/home/yi.yang/src/videoGenV1/repos/Wan2.2")
WAN_CKPT = Path("/gpfs/sjtu/audiocc/home/yi.yang/videogen/models/Wan2.2-TI2V-5B")
CLIPS = Path("/home/yi.yang/src/videoGenV2/project/deepseek/video_data_raw/clips")
REFS = {
    "正面": Path("/home/yi.yang/src/videoGenV2/project/deepseek/deepFront_9x16.jpg"),
    "侧面": Path("/home/yi.yang/src/videoGenV2/project/deepseek/deepseekSide_9x16.jpg"),
    "背面或转身": Path("/home/yi.yang/src/videoGenV2/project/deepseek/deepseekBack_9x16.jpg"),
}
FFMPEG = "/home/yi.yang/miniforge3/envs/videogen-v2/bin/ffmpeg"
FFPROBE = "/home/yi.yang/miniforge3/envs/videogen-v2/bin/ffprobe"
# These clips teach a shrinking body or a cropped one. Their sample weight
# is divided by three when --downweight is set.
DOWNWEIGHT = {
    "ref_back_01.mp4",
    "seedance_backwalk_01.mp4",
    "seedance_pier_01.mp4",
    "seedance_swing_01.mp4",
}


class LoRALinear(nn.Module):
    def __init__(self, base, rank):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.normal_(self.lora_A, std=1.0 / math.sqrt(rank))

    def forward(self, x):
        # Wan casts some activations to fp32 before the linear, while the
        # frozen base weights stay bf16. Cast explicitly and keep the LoRA
        # product attached to the fp32 parameters.
        out_dtype = x.dtype
        with torch.amp.autocast("cuda", enabled=False):
            weight = self.base.weight
            bias = self.base.bias
            x_base = x.to(dtype=weight.dtype)
            y = F.linear(x_base, weight, bias if bias is not None else None)
            x_lora = x.to(dtype=self.lora_A.dtype)
            delta = (x_lora @ self.lora_A.transpose(0, 1)) @ self.lora_B.transpose(0, 1)
            return (y + delta.to(dtype=y.dtype)).to(dtype=out_dtype)


def probe(path):
    raw = subprocess.check_output([
        FFPROBE, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate,nb_frames,duration",
        "-of", "json", str(path),
    ], text=True)
    stream = json.loads(raw)["streams"][0]
    num, den = stream["avg_frame_rate"].split("/")
    fps = float(num) / float(den)
    frames = stream.get("nb_frames")
    if frames and frames != "N/A":
        count = int(frames)
    else:
        count = int(float(stream["duration"]) * fps)
    return fps, count


def read_window(path, start, count, width, height, fps):
    # Select by frame index. Seeking with -ss rounds to a timestamp and can
    # drop the last frame of a clip.
    end = start + count - 1
    cmd = [
        FFMPEG, "-v", "error", "-i", str(path),
        "-vf", "select='between(n,%s,%s)',scale=%s:%s:flags=area" % (
            start, end, width, height),
        "-frames:v", str(count),
        "-fps_mode", "passthrough",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    raw = subprocess.check_output(cmd)
    expected = count * height * width * 3
    if len(raw) != expected:
        print("SHORT_WINDOW %s start %s fps %.3f got %s expected %s" % (
            path.name, start, fps, len(raw), expected), flush=True)
        return None
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(count, height, width, 3)
    ten = torch.from_numpy(arr.copy()).permute(3, 0, 1, 2).float()
    return ten / 127.5 - 1.0


def load_rows():
    rows = json.loads((CLIPS / "features.json").read_text(encoding="utf-8"))
    usable = []
    for row in rows:
        path = CLIPS / row["file"]
        if not path.is_file() or "caption" not in row:
            continue
        fps, count = probe(path)
        row = dict(row)
        row["path"] = path
        row["fps"] = fps
        row["nframes"] = count
        usable.append(row)
    if not usable:
        raise SystemExit("no captioned clips in %s" % CLIPS)
    return usable


def pick(rows, rng):
    weights = [r["sample_weight"] for r in rows]
    return rng.choices(rows, weights=weights, k=1)[0]


def uniform_starts(nframes, window):
    """Evenly spaced starts. Every fourth frame keeps the Wan VAE alignment."""
    if nframes < window:
        return []
    return list(range(0, nframes - window + 1, 4))


def other_index(nframes, start, window, rng, gap=16):
    """A frame of the same clip that is not this window's first frame.

    Prefer a frame at least `gap` away from the window so the pinned pose is
    not the pose the later frames continue. Fall back to any frame outside
    the window, then to a later frame inside it.
    """
    window_end = start + window
    far = [i for i in range(nframes) if i <= start - gap or i >= window_end + gap]
    if far:
        return rng.choice(far)
    outside = [i for i in range(nframes) if i < start or i >= window_end]
    if outside:
        return rng.choice(outside)
    inside = [i for i in range(start + 1, min(window_end, nframes))]
    if inside:
        return rng.choice(inside)
    return None


def reference_tensor(path, width, height):
    """Scale and center-crop one reference the way Wan's i2v preprocess does."""
    img = Image.open(path).convert("RGB")
    scale = max(width / img.width, height / img.height)
    resized = img.resize(
        (round(img.width * scale), round(img.height * scale)), Image.Resampling.LANCZOS)
    x1 = (resized.width - width) // 2
    y1 = (resized.height - height) // 2
    cropped = resized.crop((x1, y1, x1 + width, y1 + height))
    arr = np.asarray(cropped, dtype=np.float32)
    ten = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(1)
    return ten / 127.5 - 1.0


def inject_lora(model, rank, attn_only=False):
    targets = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        leaf = name.rsplit(".", 1)[-1]
        attn = leaf in {"q", "k", "v", "o"} and (
            ".self_attn." in name or ".cross_attn." in name)
        ffn = name.endswith(".ffn.0") or name.endswith(".ffn.2")
        if attn or (ffn and not attn_only):
            targets.append(name)
    for name in targets:
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        base = getattr(parent, child)
        wrapped = LoRALinear(base, rank)
        wrapped.to(device=base.weight.device)
        setattr(parent, child, wrapped)
    return targets


def checkpoint_blocks(model):
    for block in model.blocks:
        orig = block.forward

        def wrapped(x, e, seq_lens, grid_sizes, freqs, context, context_lens, _orig=orig):
            if not x.requires_grad:
                x = x.requires_grad_(True)
            return torch.utils.checkpoint.checkpoint(
                _orig, x, e, seq_lens, grid_sizes, freqs, context, context_lens,
                use_reentrant=False)

        block.forward = wrapped


def save_lora(model, path, rank):
    tensors = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            tensors["diffusion_model.%s.lora_A.weight" % name] = module.lora_A.detach().float().cpu()
            tensors["diffusion_model.%s.lora_B.weight" % name] = module.lora_B.detach().float().cpu()
    path.parent.mkdir(parents=True, exist_ok=True)
    from safetensors.torch import save_file
    save_file(tensors, str(path), metadata={"rank": str(rank), "format": "wan_diffusion_lora"})
    print("LORA_SAVED %s tensors %s" % (path, len(tensors) // 2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--frames", type=int, default=17)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=832)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--text-drop", type=float, default=0.1)
    parser.add_argument("--anchor-prob", type=float, default=0.25)
    parser.add_argument("--pin-first", action="store_true")
    parser.add_argument("--sheet-prob", type=float, default=0.1)
    parser.add_argument("--clip-ref", action="store_true")
    parser.add_argument("--ref-prob", type=float, default=0.4)
    parser.add_argument("--attn-only", action="store_true")
    parser.add_argument("--downweight", action="store_true")
    parser.add_argument("--seed", type=int, default=81)
    parser.add_argument("--save-every", type=int, default=40)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/gpfs/sjtu/audiocc/home/yi.yang/videogen-v2/lora/whale_character_r16.safetensors"),
    )
    parser.add_argument("--tier", choices=("all", "clean", "noisy"), default="all")
    parser.add_argument("--check-data", action="store_true")
    args = parser.parse_args()
    if args.frames % 4 != 1:
        raise SystemExit("--frames must be 4n+1 so the Wan VAE can encode it")
    if args.width % 32 or args.height % 32:
        raise SystemExit("width and height must be divisible by 32")
    if not 0.0 <= args.anchor_prob <= 1.0:
        raise SystemExit("--anchor-prob must be between 0 and 1")
    if not 0.0 <= args.sheet_prob <= 1.0:
        raise SystemExit("--sheet-prob must be between 0 and 1")
    if not 0.0 <= args.ref_prob <= 1.0:
        raise SystemExit("--ref-prob must be between 0 and 1")
    if args.clip_ref and args.pin_first:
        raise SystemExit("--clip-ref and --pin-first are different objectives")
    if not args.clip_ref:
        for label, path in REFS.items():
            if not path.is_file():
                raise SystemExit("missing %s reference: %s" % (label, path))

    rows = load_rows()
    if args.tier != "all":
        rows = [row for row in rows if row["tier"] == args.tier]
        if not rows:
            raise SystemExit("no clips with tier %s" % args.tier)
    clean = sum(r["seconds"] for r in rows if r["tier"] == "clean")
    noisy = sum(r["seconds"] for r in rows if r["tier"] == "noisy")
    print("CLIPS %s tier %s clean_s %.1f noisy_s %.1f" % (
        len(rows), args.tier, clean, noisy), flush=True)
    if args.downweight:
        touched = []
        for row in rows:
            if row["file"] in DOWNWEIGHT:
                row["sample_weight"] = round(float(row["sample_weight"]) / 3.0, 4)
                touched.append(row["file"])
        print("DOWNWEIGHT %s" % " ".join(touched), flush=True)
    for row in rows:
        row["starts"] = uniform_starts(row["nframes"], args.frames)
    covered = sum(1 for row in rows if row["starts"])
    short = [row["file"] for row in rows if not row["starts"]]
    print("UNIFORM_WINDOWS %s/%s clips" % (covered, len(rows)), flush=True)
    if short:
        print("SHORT %s" % " ".join(short), flush=True)
    if args.check_data:
        row = rows[0]
        video = read_window(row["path"], 0, args.frames, args.width, args.height, row["fps"])
        print("CHECK %s %s fps %.3f frames %s tensor %s" % (
            row["file"], row["tier"], row["fps"], row["nframes"], tuple(video.shape)), flush=True)
        if args.clip_ref:
            far = near = inside = missing = 0
            for item in rows:
                for start in item["starts"]:
                    window_end = start + args.frames
                    nframes = item["nframes"]
                    if any(i <= start - 16 or i >= window_end + 16 for i in range(nframes)):
                        far += 1
                    elif any(i < start or i >= window_end for i in range(nframes)):
                        near += 1
                    elif start + 1 < min(window_end, nframes):
                        inside += 1
                    else:
                        missing += 1
            print("CLIPREF far %s near %s inside %s missing %s" % (
                far, near, inside, missing), flush=True)
        return

    sys.path.insert(0, str(WAN_REPO))
    from wan.configs import WAN_CONFIGS
    from wan.modules.model import WanModel
    from wan.modules.t5 import T5EncoderModel
    from wan.modules.vae2_2 import Wan2_2_VAE

    cfg = WAN_CONFIGS["ti2v-5B"]
    device = torch.device("cuda")
    captions = sorted({r["caption"] for r in rows})
    print("T5_ENCODE %s unique captions" % len(captions), flush=True)
    t5 = T5EncoderModel(
        text_len=cfg.text_len,
        dtype=cfg.t5_dtype,
        device=torch.device("cpu"),
        checkpoint_path=str(WAN_CKPT / cfg.t5_checkpoint),
        tokenizer_path=str(WAN_CKPT / cfg.t5_tokenizer),
    )
    encoded = t5(captions + [""], torch.device("cpu"))
    cache = {text: emb.detach().cpu() for text, emb in zip(captions + [""], encoded)}
    del t5
    print("T5_FREED", flush=True)

    vae = Wan2_2_VAE(
        vae_pth=str(WAN_CKPT / cfg.vae_checkpoint),
        device=device)
    print("VAE_READY", flush=True)
    ref_latent = {}
    if not args.clip_ref:
        with torch.no_grad():
            for label, path in REFS.items():
                frame = reference_tensor(path, args.width, args.height)
                encoded = vae.encode([frame.to(device)])[0].detach()
                if encoded.shape[1] != 1:
                    raise SystemExit("reference %s encoded %s frames" % (label, encoded.shape[1]))
                ref_latent[label] = encoded
                print("REF %s %s" % (label, tuple(encoded.shape)), flush=True)
    model = WanModel.from_pretrained(str(WAN_CKPT))
    model.to(device=device, dtype=torch.bfloat16)
    model.requires_grad_(False)
    model.train()
    names = inject_lora(model, args.rank, attn_only=args.attn_only)
    checkpoint_blocks(model)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print("LORA_LAYERS %s trainable_params %s" % (
        len(names), sum(p.numel() for p in trainable)), flush=True)
    opt = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.99), weight_decay=0.0)
    rng = random.Random(args.seed)
    log_path = args.out.with_suffix(".jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    patch_t, patch_h, patch_w = cfg.patch_size

    for step in range(1, args.steps + 1):
        t0 = time.time()
        video = None
        for _ in range(20):
            row = pick(rows, rng)
            if not row["starts"]:
                print("SKIP_SHORT %s" % row["file"], flush=True)
                continue
            start = rng.choice(row["starts"])
            video = read_window(
                row["path"], start, args.frames, args.width, args.height, row["fps"])
            if video is not None:
                break
        if video is None:
            raise SystemExit("no full %s-frame window at step %s" % (args.frames, step))
        with torch.no_grad():
            latent = vae.encode([video.to(device)])[0].detach()
        _, frames, height, width = latent.shape
        tokens_per_frame = (height // patch_h) * (width // patch_w)
        n_tok = frames * tokens_per_frame
        ref_frame = None
        if args.clip_ref:
            sheet = False
            pin = "none"
            if rng.random() < args.ref_prob:
                ref_frame = other_index(row["nframes"], start, args.frames, rng)
                frame = None if ref_frame is None else read_window(
                    row["path"], ref_frame, 1, args.width, args.height, row["fps"])
                if frame is not None:
                    with torch.no_grad():
                        ref = vae.encode([frame.to(device)])[0].detach()
                    same = (
                        ref.shape[1] == 1
                        and ref.shape[0] == latent.shape[0]
                        and ref.shape[2:] == latent.shape[2:])
                    if same:
                        latent = latent.clone()
                        latent[:, :1] = ref
                        pin = "clipref"
                    else:
                        print("REF_SHAPE %s %s vs %s" % (
                            row["file"], tuple(ref.shape), tuple(latent.shape)), flush=True)
                        ref_frame = None
                else:
                    ref_frame = None
        elif args.pin_first:
            sheet = rng.random() < args.sheet_prob
            pin = "sheet" if sheet else "clip"
        else:
            sheet = rng.random() < args.anchor_prob
            pin = "sheet" if sheet else "none"
        if sheet:
            label = row.get("view", "正面")
            if label not in ref_latent:
                label = "正面"
            latent = latent.clone()
            latent[:, :1] = ref_latent[label]
        noise = torch.randn_like(latent)
        u = torch.rand((), device=device)
        sigma = args.shift * u / (1 + (args.shift - 1) * u)
        sigma = sigma.clamp(1e-4, 1 - 1e-4)
        noisy = (1 - sigma) * latent + sigma * noise
        if pin != "none":
            noisy[:, :1] = latent[:, :1]
        target = noise - latent
        t_token = torch.full((1, n_tok), float(sigma * 1000), device=device)
        if pin != "none":
            t_token[:, :tokens_per_frame] = 0
        text = "" if rng.random() < args.text_drop else row["caption"]
        context = cache[text].to(device=device, dtype=torch.bfloat16)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pred = model([noisy.to(dtype=torch.bfloat16)], t_token, [context], n_tok)[0]
        err = (pred.float() - target.float()).pow(2)
        if pin != "none":
            err = err[:, 1:]
        loss = err.mean()
        if not torch.isfinite(loss):
            raise SystemExit("non-finite loss at step %s" % step)
        loss.backward()
        grad_ok = any(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable)
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        record = {
            "step": step,
            "loss": round(float(loss.detach().cpu()), 6),
            "sigma": round(float(sigma.detach().cpu()), 4),
            "file": row["file"],
            "tier": row["tier"],
            "pin": pin,
            "ref_frame": ref_frame,
            "view": row.get("view", ""),
            "seconds": round(time.time() - t0, 2),
            "grad": grad_ok,
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print("STEP %s" % json.dumps(record, ensure_ascii=False), flush=True)
        if step == 1 and not grad_ok:
            raise SystemExit("LoRA received no gradient on step 1")
        if step % args.save_every == 0 or step == args.steps:
            ckpt = args.out.with_name("%s_s%04d%s" % (args.out.stem, step, args.out.suffix))
            save_lora(model, ckpt, args.rank)


if __name__ == "__main__":
    main()
