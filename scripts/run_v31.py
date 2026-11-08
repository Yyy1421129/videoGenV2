"""One AniSora V3.1 12GB clip through the bundled LightX2V FP8 runtime.

The wan.7z package is not the full bf16 checkpoint. It is a Windows LightX2V
build: FP8 blocks, 640x360, 49 frames, 8 steps, block offload. This script
keeps that recipe and only rewrites Windows paths. No lip-sync.
"""
import json
import os
import sys
from pathlib import Path

WAN = Path("/gpfs/sjtu/audiocc/home/yi.yang/videogen-v2/models/anisora-v3.1-12g/extracted/wan")
STUBS = Path("/home/yi.yang/src/videoGenV2/stubs")
sys.path.insert(0, str(WAN))
sys.path.insert(0, str(STUBS))
os.environ.setdefault("DTYPE", "BF16")
os.environ.setdefault("ENABLE_GRAPH_MODE", "false")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from lightx2v.infer import init_runner
from lightx2v.utils.set_config import set_config


def build_config(shot: Path, spec: dict) -> Path:
    src = json.loads((WAN / "anisora-fp8.json").read_text(encoding="utf-8"))
    model = WAN / "models" / "anisora"
    src["model_path"] = str(model)
    src["dit_quantized_ckpt"] = str(model / "fp8")
    src["clip_quantized_ckpt"] = str(WAN / "models" / "clip-fp8.pth")
    src["t5_quantized_ckpt"] = None
    for key in ("self_attn_1_type", "cross_attn_1_type", "cross_attn_2_type"):
        src[key] = "flash_attn2"
    src["seed"] = int(spec.get("seed", 42))
    prompt = spec["wan_prompt"].strip()
    if "motion score" not in prompt.lower():
        prompt += " aesthetic score: 5.5. motion score: 3.0. There is no text in the video."
    n_prompt = spec.get("n_prompt") or ""
    for word in ("风格", "作品", "画作", "画面"):
        n_prompt = n_prompt.replace(word, "")
    src["prompt"] = prompt
    src["negative_prompt"] = n_prompt
    src["image_path"] = str(shot / "candidates" / "1" / "frame0.png")
    out = shot / "candidates" / "v31" / "wan.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    src["save_video_path"] = str(out)
    path = shot / "candidates" / "v31" / "anisora-fp8.json"
    path.write_text(json.dumps(src, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


class Args:
    def __init__(self, shot: Path, spec: dict, config_json: Path):
        self.model_cls = "wan2.1"
        self.task = "i2v"
        self.model_path = str(WAN / "models" / "anisora")
        self.config_json = str(config_json)
        self.use_prompt_enhancer = False
        self.prompt = json.loads(config_json.read_text(encoding="utf-8"))["prompt"]
        self.negative_prompt = json.loads(config_json.read_text(encoding="utf-8"))["negative_prompt"]
        self.image_path = str(shot / "candidates" / "1" / "frame0.png")
        self.last_frame_path = ""
        self.audio_path = ""
        self.src_ref_images = None
        self.src_video = None
        self.src_mask = None
        self.save_video_path = str(shot / "candidates" / "v31" / "wan.mp4")


def main() -> None:
    shot = Path(sys.argv[sys.argv.index("--shot") + 1]).resolve()
    spec = json.loads((shot / "shot.json").read_text(encoding="utf-8"))
    first = shot / "candidates" / "1" / "frame0.png"
    if not first.is_file():
        raise SystemExit("missing conditioning frame: %s" % first)
    out = shot / "candidates" / "v31" / "wan.mp4"
    if out.is_file() and out.stat().st_size > 0:
        print("SKIP", out, flush=True)
        return
    config_json = build_config(shot, spec)
    config = set_config(Args(shot, spec, config_json))
    runner = init_runner(config)
    runner.run_pipeline()
    if not out.is_file() or out.stat().st_size == 0:
        raise SystemExit("V3.1 did not write %s" % out)
    print("V31_OK", out, flush=True)


if __name__ == "__main__":
    main()
