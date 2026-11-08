#!/usr/bin/env python3
"""Cut whale-girl clips without watching the video.

Samples each clip at 4 fps and keeps a frame when three checks pass:
the MediaPipe hand box is no bigger than her own chibi hands, the lower
half has no outlined subtitle, and a detected face has blue rather than
purple eyes. Consecutive kept samples become one clip. A single bad
sample does not split a clip. Segments shorter than 2 seconds are dropped.

    python scripts/slice_whale_clips.py project/deepseek/video_data_raw/nowm
"""

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

SAMPLE_FPS = 4
MIN_SECONDS = 2.0
HAND_BOX = 70  # longer side, on a 360-wide frame; her hands are about 50
FFMPEG = "/home/yi.yang/miniforge3/envs/videogen-v2/bin/ffmpeg"


def subtitle(bgr):
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr[int(h * 0.55):], cv2.COLOR_BGR2GRAY)
    bright = gray > 210
    dark = cv2.dilate((gray < 80).astype(np.uint8), np.ones((3, 3), np.uint8))
    row = (bright & (dark > 0)).sum(axis=1)
    if row.max() < w * 0.12:
        return False
    band = row > w * 0.05
    best = cur = 0
    for value in band:
        cur = cur + 1 if value else 0
        best = max(best, cur)
    return best >= 8


def photo_hand(bgr, result):
    if not result.multi_hand_landmarks:
        return False
    h, w = bgr.shape[:2]
    for landmarks in result.multi_hand_landmarks:
        xs = [point.x for point in landmarks.landmark]
        ys = [point.y for point in landmarks.landmark]
        longer = max((max(xs) - min(xs)) * w, (max(ys) - min(ys)) * h)
        if longer >= HAND_BOX:
            return True
    return False


def other_character(bgr, result):
    if not result.detections:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        blue = (hsv[:, :, 0] > 95) & (hsv[:, :, 0] < 130) & (hsv[:, :, 1] > 80)
        return float(blue.mean()) < 0.06
    h, w = bgr.shape[:2]
    det = max(result.detections, key=lambda item: item.location_data.relative_bounding_box.width)
    box = det.location_data.relative_bounding_box
    x = int(box.xmin * w)
    y = int(box.ymin * h)
    bw = int(box.width * w)
    bh = int(box.height * h)
    patch = bgr[y + int(bh * 0.25):y + int(bh * 0.55), x + int(bw * 0.2):x + int(bw * 0.8)]
    if patch.size == 0:
        return True
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    keep = (hsv[:, :, 1] > 60) & (hsv[:, :, 2] > 40)
    if keep.sum() < 80:
        return True
    hue = hsv[:, :, 0][keep]
    blue = ((hue >= 90) & (hue < 125)).mean()
    purple = ((hue >= 125) & (hue < 165)).mean()
    return blue < 0.45 or purple > 0.40


def segments_from(flags, step):
    """flags[i] is bad. Ignore isolated bad samples."""
    bad = flags[:]
    for i, value in enumerate(bad):
        if value and not (i and bad[i - 1]) and not (i + 1 < len(bad) and flags[i + 1]):
            bad[i] = False
    spans = []
    start = None
    for i, value in enumerate(bad):
        if not value and start is None:
            start = i
        if value and start is not None:
            spans.append((start * step, i * step))
            start = None
    if start is not None:
        spans.append((start * step, len(bad) * step))
    return [(a, b) for a, b in spans if b - a >= MIN_SECONDS]


def scan(path, hands, face):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    step = max(1, int(round(fps / SAMPLE_FPS)))
    flags = []
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index % step == 0:
            small = cv2.resize(frame, (360, 640))
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            bad = (
                photo_hand(small, hands.process(rgb))
                or subtitle(small)
                or other_character(small, face.process(rgb))
            )
            flags.append(bad)
        index += 1
    cap.release()
    duration = index / fps
    spans = segments_from(flags, step / fps)
    spans = [(a, min(b, duration)) for a, b in spans]
    return spans


def cut(src, spans, dest, ffmpeg):
    written = []
    for number, (start, end) in enumerate(spans, 1):
        out = dest / f"{src.stem}_{number:02d}.mp4"
        subprocess.check_call([
            ffmpeg, "-y", "-v", "error",
            "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
            "-i", str(src),
            "-c:v", "libx264", "-crf", "18", "-preset", "fast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "144k",
            str(out),
        ])
        written.append({"file": out.name, "start": round(start, 3), "end": round(end, 3)})
    return written


def main():
    parser = argparse.ArgumentParser(description="Slice whale-girl clips on the server.")
    parser.add_argument("path", type=Path, help="Directory of watermark-free mp4 files.")
    parser.add_argument("--out", type=Path, help="Output directory. Defaults to path/../clips.")
    args = parser.parse_args()
    source = args.path.resolve()
    dest = (args.out or source.parent / "clips").resolve()
    dest.mkdir(parents=True, exist_ok=True)
    hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.4)
    face = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.4)
    report = []
    for video in sorted(source.glob("train_data*.mp4")):
        spans = scan(video, hands, face)
        clips = cut(video, spans, dest, FFMPEG)
        report.append({"source": video.name, "clips": clips})
        print(video.name, ", ".join(f"{item['start']}-{item['end']}" for item in clips) or "none")
    hands.close()
    face.close()
    (dest / "clips.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
