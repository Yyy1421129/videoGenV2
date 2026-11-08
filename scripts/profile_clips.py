#!/usr/bin/env python3
"""Measure motion, background, and layout of every clip in clips.json.

Writes features.json next to the manifest. Clustering uses these numbers;
semantic labels are filled in by hand in the same file.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

CLIPS = Path("/home/yi.yang/src/videoGenV2/project/deepseek/video_data_raw/clips")
MANIFEST = CLIPS / "clips.json"
OUT = CLIPS / "features.json"


def stats(path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(round(fps / 2)))
    frames = []
    for i in range(0, n, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(frame, (90, 160), interpolation=cv2.INTER_AREA))
    cap.release()
    if len(frames) < 2:
        return None
    arr = np.stack(frames).astype(np.float32)
    diff = np.abs(arr[1:] - arr[:-1]).mean(axis=-1)
    motion = float(diff.mean())
    upper = float(diff[:, :64].mean())
    lower = float(diff[:, 96:].mean())
    white = float((arr.mean(axis=-1) > 235).mean())
    hsv = cv2.cvtColor(arr[len(arr) // 2].astype(np.uint8), cv2.COLOR_BGR2HSV)
    sat = float(hsv[:, :, 1].mean())
    # non-white mass: where the character or the scene sits vertically
    gray = arr.mean(axis=-1)
    ink = gray < 220
    rows = ink.mean(axis=(0, 2))
    if rows.sum() < 1e-3:
        centroid = 0.5
    else:
        centroid = float((rows * np.arange(rows.shape[0])).sum() / rows.sum() / rows.shape[0])
    return {
        "samples": len(frames),
        "motion": round(motion, 3),
        "motion_upper": round(upper, 3),
        "motion_lower": round(lower, 3),
        "white_fraction": round(white, 3),
        "saturation": round(sat, 1),
        "vertical_centroid": round(centroid, 3),
    }


def main():
    groups = json.loads(MANIFEST.read_text())
    rows = []
    for group in groups:
        tier = group.get("tier", "clean")
        for clip in group["clips"]:
            path = CLIPS / clip["file"]
            measured = stats(path)
            if measured is None:
                print("skip", clip["file"], file=sys.stderr)
                continue
            span = round(clip["end"] - clip["start"], 3)
            rows.append({
                "file": clip["file"],
                "source": group["source"],
                "tier": tier,
                "seconds": span,
                **measured,
            })
            print(clip["file"], measured["motion"], measured["white_fraction"], flush=True)
    # 4-means on motion, white, saturation, centroid, lower/upper ratio
    feat = np.array([
        [
            r["motion"],
            r["white_fraction"],
            r["saturation"] / 100.0,
            r["vertical_centroid"],
            r["motion_lower"] / max(r["motion_upper"], 1e-3),
        ]
        for r in rows
    ], dtype=np.float64)
    mu = feat.mean(axis=0)
    sd = feat.std(axis=0)
    sd[sd < 1e-6] = 1
    z = (feat - mu) / sd
    rng = np.random.default_rng(0)
    centers = z[rng.choice(len(z), size=4, replace=False)]
    labels = np.zeros(len(z), dtype=int)
    for _ in range(30):
        dist = ((z[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = dist.argmin(axis=1)
        for k in range(4):
            if (labels == k).any():
                centers[k] = z[labels == k].mean(axis=0)
    for row, label in zip(rows, labels.tolist()):
        row["cluster"] = int(label)
    OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    print("wrote", OUT, "clips", len(rows))


if __name__ == "__main__":
    main()
