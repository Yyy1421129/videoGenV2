#!/usr/bin/env python3
"""CosyVoice once, then Wan2.2 TI2V-5B with the anime LoRA baked in.

The Wan repository is imported and not modified. Lip-sync is intentionally
not run in this stage.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import torch
from PIL import Image

WAN_REPO = Path("/home/yi.yang/src/videoGenV1/repos/Wan2.2")
sys.path.insert(0, str(WAN_REPO))

import wan
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, WAN_CONFIGS
from wan.utils.utils import save_video


PY = sys.executable
FFMPEG = os.environ.get("FFMPEG", "/home/yi.yang/miniforge3/envs/videogen-v2/bin/ffmpeg")
WAN_CKPT = "/gpfs/sjtu/audiocc/home/yi.yang/videogen/models/Wan2.2-TI2V-5B"
TTS = Path("/home/yi.yang/src/videoGenV2/scripts/tts_one.py")


def nonempty(path):
    return path.is_file() and path.stat().st_size > 0


def run(cmd, cwd):
    print("RUN", " ".join(cmd), flush=True)
    code = subprocess.call(cmd, cwd=str(cwd))
    if code != 0:
        raise SystemExit(code)


def apply_lora(model, path, scale):
    from safetensors.torch import load_file
    tensors = load_file(path)
    grouped = {}
    for key, value in tensors.items():
        if not key.startswith("diffusion_model."):
            continue
        if key.endswith(".lora_A.weight"):
            name = key[len("diffusion_model."):-len(".lora_A.weight")]
            grouped.setdefault(name, {})["A"] = value
        elif key.endswith(".lora_B.weight"):
            name = key[len("diffusion_model."):-len(".lora_B.weight")]
            grouped.setdefault(name, {})["B"] = value
    applied = 0
    missing = []
    with torch.no_grad():
        for name, pair in grouped.items():
            if "A" not in pair or "B" not in pair:
                continue
            try:
                module = model.get_submodule(name)
            except AttributeError:
                missing.append(name)
                continue
            update = pair["B"].float() @ pair["A"].float()
            module.weight.add_(update.to(dtype=module.weight.dtype, device=module.weight.device) * scale)
            applied += 1
    print("LORA_APPLIED %s scale %s missing %s" % (applied, scale, len(missing)), flush=True)
    if applied == 0:
        raise SystemExit("anime LoRA matched no Wan layers: %s" % path)
    if missing:
        print("LORA_MISSING", " ".join(missing[:8]), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shot", required=True, type=Path)
    args = parser.parse_args()
    shot = args.shot
    spec = json.loads((shot / "shot.json").read_text(encoding="utf-8"))
    if not Path(spec["lora"]).is_file():
        raise SystemExit("lora file missing: %s" % spec["lora"])
    seeds = [int(s) for s in spec.get("seeds") or [spec.get("seed", 42)]]
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

    voice = shot / "voice.wav"
    if nonempty(voice):
        print("reuse", voice)
    else:
        run([PY, str(TTS), "--shot", str(shot)], cwd=shot)

    pending = []
    for index, seed in enumerate(seeds, start=1):
        cand = shot / "candidates" / str(index)
        cand.mkdir(parents=True, exist_ok=True)
        silent = cand / "wan.mp4"
        if nonempty(silent):
            print("reuse", silent)
        else:
            pending.append((index, seed, silent))
    if not pending:
        print("SHOT_OK", shot / "candidates")
        return

    cfg = WAN_CONFIGS["ti2v-5B"]
    device = int(os.environ.get("LOCAL_RANK", "0"))
    pipe = wan.WanTI2V(
        config=cfg,
        checkpoint_dir=WAN_CKPT,
        device_id=device,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=True,
        convert_model_dtype=True,
    )
    apply_lora(pipe.model, spec["lora"], float(spec.get("lora_scale", 1.0)))
    image = None
    if spec.get("text_only"):
        print("TEXT_ONLY", flush=True)
    else:
        image_path = spec.get("image")
        if not image_path:
            view = spec.get("view", "front")
            image_path = spec.get("image_back" if view == "back" else "image_front")
        if image_path:
            image = Image.open(image_path).convert("RGB")
            print("IMAGE", image_path, image.size, flush=True)

    for index, seed, silent in pending:
        logging.info("candidate %s seed %s", index, seed)
        video = pipe.generate(
            spec["wan_prompt"],
            img=image,
            size=SIZE_CONFIGS[spec["size"]],
            max_area=MAX_AREA_CONFIGS[spec["size"]],
            frame_num=int(spec["frame_num"]),
            shift=cfg.sample_shift,
            sample_solver="unipc",
            sampling_steps=int(spec["sample_steps"]),
            guide_scale=float(spec.get("guide_scale", cfg.sample_guide_scale)),
            seed=seed,
            offload_model=True,
            n_prompt=spec["n_prompt"],
        )
        save_video(
            tensor=video[None],
            save_file=str(silent),
            fps=cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )
        del video
        torch.cuda.empty_cache()
        voiced = shot / "candidates" / str(index) / "voiced.mp4"
        if nonempty(voice) and not nonempty(voiced):
            subprocess.call([
                FFMPEG, "-y", "-i", str(silent), "-i", str(voice),
                "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                str(voiced),
            ])
        print("CANDIDATE_OK", silent, flush=True)
    print("SHOT_OK", shot / "candidates", flush=True)


if __name__ == "__main__":
    main()
