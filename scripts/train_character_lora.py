"""Character LoRA for Wan2.2 TI2V-5B.

Freezes the DiT, the T5 encoder, and the VAE. Trains rank-16 LoRA on the
attention and feed-forward linears with flow-matching loss. The first latent
frame stays clean and is left out of the loss, matching TI2V inference.

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

WAN_REPO = Path("/home/yi.yang/src/videoGenV1/repos/Wan2.2")
WAN_CKPT = Path("/gpfs/sjtu/audiocc/home/yi.yang/videogen/models/Wan2.2-TI2V-5B")
CLIPS = Path("/home/yi.yang/src/videoGenV2/project/deepseek/video_data_raw/clips")
FFMPEG = "/home/yi.yang/miniforge3/envs/videogen-v2/bin/ffmpeg"
FFPROBE = "/home/yi.yang/miniforge3/envs/videogen-v2/bin/ffprobe"


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
        y = F.linear(x, self.base.weight, self.base.bias)
        a = self.lora_A.to(dtype=x.dtype)
        b = self.lora_B.to(dtype=x.dtype)
        return y + (x @ a.transpose(0, 1) @ b.transpose(0, 1))


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
    cmd = [
        FFMPEG, "-v", "error", "-i", str(path),
        "-ss", f"{start / fps:.4f}",
        "-frames:v", str(count),
        "-vf", f"scale={width}:{height}:flags=area",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    raw = subprocess.check_output(cmd)
    expected = count * height * width * 3
    if len(raw) != expected:
        raise RuntimeError("%s window %s got %s bytes, expected %s" % (
            path.name, start, len(raw), expected))
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


def inject_lora(model, rank):
    targets = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        leaf = name.rsplit(".", 1)[-1]
        attn = leaf in {"q", "k", "v", "o"} and (
            ".self_attn." in name or ".cross_attn." in name)
        ffn = name.endswith(".ffn.0") or name.endswith(".ffn.2")
        if attn or ffn:
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
    parser.add_argument("--seed", type=int, default=81)
    parser.add_argument("--save-every", type=int, default=40)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/gpfs/sjtu/audiocc/home/yi.yang/videogen-v2/lora/whale_character_r16.safetensors"),
    )
    parser.add_argument("--check-data", action="store_true")
    args = parser.parse_args()
    if args.frames % 4 != 1:
        raise SystemExit("--frames must be 4n+1 so the Wan VAE can encode it")
    if args.width % 32 or args.height % 32:
        raise SystemExit("width and height must be divisible by 32")

    rows = load_rows()
    clean = sum(r["seconds"] for r in rows if r["tier"] == "clean")
    noisy = sum(r["seconds"] for r in rows if r["tier"] == "noisy")
    print("CLIPS %s clean_s %.1f noisy_s %.1f" % (len(rows), clean, noisy), flush=True)
    if args.check_data:
        row = rows[0]
        video = read_window(row["path"], 0, args.frames, args.width, args.height, row["fps"])
        print("CHECK %s %s fps %.3f frames %s tensor %s" % (
            row["file"], row["tier"], row["fps"], row["nframes"], tuple(video.shape)), flush=True)
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
    model = WanModel.from_pretrained(str(WAN_CKPT))
    model.to(device=device, dtype=torch.bfloat16)
    model.requires_grad_(False)
    model.train()
    names = inject_lora(model, args.rank)
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
        row = pick(rows, rng)
        start_max = row["nframes"] - args.frames
        if start_max < 0:
            print("SKIP_SHORT %s" % row["file"], flush=True)
            continue
        start = rng.randint(0, start_max)
        video = read_window(row["path"], start, args.frames, args.width, args.height, row["fps"])
        with torch.no_grad():
            latent = vae.encode([video.to(device)])[0].detach()
        _, frames, height, width = latent.shape
        tokens_per_frame = (height // patch_h) * (width // patch_w)
        n_tok = frames * tokens_per_frame
        noise = torch.randn_like(latent)
        u = torch.rand((), device=device)
        sigma = args.shift * u / (1 + (args.shift - 1) * u)
        sigma = sigma.clamp(1e-4, 1 - 1e-4)
        noisy = (1 - sigma) * latent + sigma * noise
        noisy[:, 0] = latent[:, 0]
        target = noise - latent
        t_token = torch.full((1, n_tok), float(sigma * 1000), device=device)
        t_token[:, :tokens_per_frame] = 0
        text = "" if rng.random() < args.text_drop else row["caption"]
        context = cache[text].to(device=device, dtype=torch.bfloat16)
        opt.zero_grad(set_to_none=True)
        pred = model([noisy], t_token, [context], n_tok)[0]
        mask = torch.ones_like(pred)
        mask[:, 0] = 0
        loss = ((pred - target) * mask).pow(2).sum() / mask.sum().clamp(min=1)
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
            "seconds": round(time.time() - t0, 2),
            "grad": grad_ok,
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print("STEP %s" % json.dumps(record, ensure_ascii=False), flush=True)
        if step == 1 and not grad_ok:
            raise SystemExit("LoRA received no gradient on step 1")
        if step % args.save_every == 0 or step == args.steps:
            save_lora(model, args.out, args.rank)


if __name__ == "__main__":
    main()
