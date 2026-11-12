#!/usr/bin/env python3
"""Plan one whale-girl anime shot from a theme prompt.

The character images and identity text stay fixed. Deepseek only fills the
line, the scene, the single action, and whether the back view is required.
Wan receives one reference image per shot.
"""

import argparse
import json
import re
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path


SECRET = Path("/home/yi.yang/src/secrets/SJTU_api.txt")
ROOT = Path("/home/yi.yang/src/videoGenV2")
PROJECT = ROOT / "project"
SHOTS = ROOT / "shots"
CHARACTER = PROJECT / "deepseek" / "character.json"
PROMPT_WAV = "/home/yi.yang/src/CosyVoice/asset/zero_shot_prompt.wav"
PROMPT_TEXT = "希望你以后能够做的比我还好呦。"
LORA = "/gpfs/sjtu/audiocc/home/yi.yang/videogen-v2/models/lora/wan_2.2_B_anime_v01_000153000.safetensors"
MODELS = {
    "deepseek": "deepseek-chat",
    "qwen": "qwen",
    "deepseek-chat": "deepseek-chat",
    "deepseek-reasoner": "deepseek-reasoner",
    "qwen3.8-27b": "qwen3.8-27b",
}

SYSTEM = """你在为固定角色「Deepseek 鲸鱼娘」写一个 5 秒动漫镜头。角色的脸、蓝发、女仆装和鲸鱼围裙已经由参考图决定，不要重新设计角色，不要写成近景或只剩头肩。

只输出一个 JSON 对象，不要 Markdown。字段：
- tts_text：一句中文台词，8 到 22 个字，没有数字、英文和舞台说明。必须能在 5 秒内念完。
- scene_en：英文场景，只写背景、光线和地面。不要描写角色外貌。
- action_en：英文，只有一个主动作。写明这个动作在大约一秒内逐帧发生，然后保持到结尾。不要同时要求两个大幅度动作。
- view：只能是 front 或 back。角色背对观众、展示头发或裙摆时用 back，其余用 front。
"""


def load_secret(path):
    vals = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip().strip(",")
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        vals[key.strip().strip('"')] = value.strip().strip('"').strip(",")
    if "baseUrl" not in vals or "apiKey" not in vals:
        raise SystemExit("secret file must contain baseUrl and apiKey")
    return vals["baseUrl"].rstrip("/"), vals["apiKey"]


def chat(base, key, model, user):
    body = json.dumps({
        "model": model,
        "temperature": 0.4,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ],
    }).encode("utf-8")
    req = urllib.request.Request(
        base + "/chat/completions",
        data=body,
        headers={
            "Authorization": "Bearer " + key,
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            payload = json.load(resp)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise SystemExit("API HTTP %s: %s" % (exc.code, detail))
    return payload["choices"][0]["message"]["content"]


def parse_json(text):
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise


def choose_view(plan, override):
    if override in ("front", "back"):
        return override
    view = str(plan.get("view", "front")).strip().lower()
    if view not in ("front", "back"):
        raise SystemExit("model view must be front or back, got %r" % plan.get("view"))
    return view


def build_prompt(character, view, scene_en, action_en):
    identity = character["identity_back" if view == "back" else "identity_front"]
    parts = [identity, scene_en.strip(), action_en.strip(), character["stillness"]]
    return " ".join(part for part in parts if part)


def main():
    parser = argparse.ArgumentParser(description="Plan one whale-girl shot from a theme.")
    parser.add_argument("--description", required=True, help="Theme or action, in Chinese or English.")
    parser.add_argument("--scene", default="白底，角色全身留在画面里")
    parser.add_argument("--style", default="与参考图一致的干净赛璐璐")
    parser.add_argument("--view", default="auto", choices=("auto", "front", "back"))
    parser.add_argument("--model", default="deepseek", choices=sorted(MODELS))
    parser.add_argument("--size", default="704*1280", choices=("1280*704", "704*1280"))
    parser.add_argument("--frame-num", type=int, default=121)
    parser.add_argument("--seed", type=int, default=80)
    parser.add_argument("--num-candidates", type=int, default=1)
    parser.add_argument("--sample-steps", type=int, default=40)
    parser.add_argument("--guide-scale", type=float, default=6.0)
    parser.add_argument("--lora-scale", type=float, default=0.6)
    parser.add_argument("--shot", default="")
    args = parser.parse_args()
    if args.frame_num < 5 or (args.frame_num - 1) % 4 != 0:
        raise SystemExit("frame_num must be 4n+1 and at least 5")
    if args.num_candidates < 1:
        raise SystemExit("num_candidates must be at least 1")

    character = json.loads(CHARACTER.read_text(encoding="utf-8"))
    for key in ("image_front", "image_back"):
        if not Path(character[key]).is_file():
            raise SystemExit("missing character image: %s" % character[key])

    shot = args.shot or datetime.now().strftime("whale-%Y%m%d-%H%M%S")
    shot_dir = SHOTS / shot
    shot_dir.mkdir(parents=True, exist_ok=False)

    base, key = load_secret(SECRET)
    raw = chat(
        base, key, MODELS[args.model],
        "主题：%s\n场景：%s\n风格：%s" % (args.description, args.scene, args.style),
    )
    plan = parse_json(raw)
    for field in ("tts_text", "scene_en", "action_en"):
        if not str(plan.get(field, "")).strip():
            raise SystemExit("model response missing %s:\n%s" % (field, raw))
    view = choose_view(plan, args.view)
    image = character["image_back" if view == "back" else "image_front"]

    spec = {
        "shot": shot,
        "character_id": character["character_id"],
        "description": args.description,
        "style": args.style,
        "scene": args.scene,
        "view": view,
        "model": MODELS[args.model],
        "tts_text": plan["tts_text"].strip(),
        "scene_en": plan["scene_en"].strip(),
        "action_en": plan["action_en"].strip(),
        "wan_prompt": build_prompt(character, view, plan["scene_en"], plan["action_en"]),
        "n_prompt": character["n_prompt_back" if view == "back" else "n_prompt_front"],
        "image": image,
        "image_front": character["image_front"],
        "image_back": character["image_back"],
        "size": args.size,
        "frame_num": args.frame_num,
        "sample_steps": args.sample_steps,
        "guide_scale": args.guide_scale,
        "seed": args.seed,
        "num_candidates": args.num_candidates,
        "seeds": [args.seed + i for i in range(args.num_candidates)],
        "lora": LORA,
        "lora_scale": args.lora_scale,
        "prompt_wav": PROMPT_WAV,
        "prompt_text": PROMPT_TEXT,
    }
    (shot_dir / "shot.json").write_text(
        json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    submit = "sbatch /home/yi.yang/slurm/videogen-v2-lora.sbatch %s" % shot_dir
    (shot_dir / "commands.sh").write_text(
        "#!/bin/bash\n# Whale-girl theme shot. Not submitted automatically.\n%s\n" % submit,
        encoding="utf-8",
    )
    print(json.dumps(
        {k: spec[k] for k in ("shot", "view", "image", "tts_text", "wan_prompt", "seeds")},
        ensure_ascii=False, indent=2))
    print("\nSHOT_DIR %s" % shot_dir)
    print(submit)


if __name__ == "__main__":
    main()
