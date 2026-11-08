#!/usr/bin/env python3
"""Screen Wan clips for a single-frame pose jump.

The score uses the same reduced frame size as the wave review. Candidate 2
peaked at 1.8; candidate 3 peaked at 22.7 on frame 9 and then held still.
"""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np


WIDTH = 92
HEIGHT = 148
SNAP_MAX = 4.0


def load_frames(path):
    raw = subprocess.check_output([
        "ffmpeg", "-v", "error", "-i", str(path),
        "-vf", "scale=%d:%d" % (WIDTH, HEIGHT),
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ])
    frame = WIDTH * HEIGHT * 3
    count = len(raw) // frame
    if count < 2:
        raise SystemExit("too few frames: %s" % path)
    arr = np.frombuffer(raw[:count * frame], dtype=np.uint8)
    return arr.reshape(count, HEIGHT, WIDTH, 3).astype(np.float32)


def summarize(path):
    video = load_frames(path)
    delta = np.mean(np.abs(video[1:] - video[:-1]), axis=(1, 2, 3))
    peak = int(np.argmax(delta)) + 1
    quarters = [float(part.mean()) for part in np.array_split(delta, 4)]
    return {
        "file": str(path),
        "frames": int(video.shape[0]),
        "mean": round(float(delta.mean()), 2),
        "max": round(float(delta.max()), 2),
        "peak_frame": peak,
        "quarters": [round(value, 2) for value in quarters],
        "snap": bool(delta.max() >= SNAP_MAX),
    }


def main():
    parser = argparse.ArgumentParser(description="Score frame-to-frame jumps in a shot.")
    parser.add_argument("--shot", required=True, type=Path)
    args = parser.parse_args()
    videos = sorted((args.shot / "candidates").glob("*/wan.mp4"))
    if not videos:
        raise SystemExit("no wan.mp4 under %s" % (args.shot / "candidates"))
    rows = [summarize(path) for path in videos]
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    kept = [row["file"] for row in rows if not row["snap"]]
    print("KEEP %s / %s" % (len(kept), len(rows)))


if __name__ == "__main__":
    main()
