#!/usr/bin/env python3
"""Synthesize one shot line with the existing CosyVoice2 weights."""

import argparse
import json
import sys
from pathlib import Path

import soundfile as sf
import torch

sys.path.insert(0, "/home/yi.yang/src/CosyVoice")
sys.path.insert(0, "/home/yi.yang/src/CosyVoice/third_party/Matcha-TTS")

from cosyvoice.cli.cosyvoice import CosyVoice2


MODEL = "/gpfs/sjtu/audiocc/home/yi.yang/videogen/models/CosyVoice2-0.5B"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shot", required=True, type=Path)
    args = parser.parse_args()
    spec = json.loads((args.shot / "shot.json").read_text(encoding="utf-8"))
    out = args.shot / "voice.wav"
    cosy = CosyVoice2(MODEL, load_jit=False, load_trt=False, fp16=False)
    chunks = []
    for item in cosy.inference_zero_shot(
        spec["tts_text"],
        spec["prompt_text"],
        spec["prompt_wav"],
        stream=False,
        text_frontend=False,
    ):
        chunks.append(item["tts_speech"])
    audio = torch.cat(chunks, dim=1).squeeze(0).cpu().numpy()
    sf.write(out, audio, cosy.sample_rate)
    print("TTS_OK %s sr %s sec %.2f" % (out, cosy.sample_rate, audio.shape[0] / cosy.sample_rate))


if __name__ == "__main__":
    main()
