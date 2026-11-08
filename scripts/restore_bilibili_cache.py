#!/usr/bin/env python3
"""Restore Bilibili client caches to mp4.

Each cache is a directory with videoInfo.json and one or more *.m4s files.
The client prefixes every fragment with the nine bytes b"000000000".
Video and audio stay in separate fragments; this script strips that prefix
and muxes them without re-encoding.

    python scripts/restore_bilibili_cache.py project/deepseek
    python scripts/restore_bilibili_cache.py project/deepseek/42034333517

The mp4 is written next to the cache directory, named from the episode title.
An existing file of the same name is left in place.
"""

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


PREFIX = b"000000000"


def ffmpeg_bin():
    found = shutil.which("ffmpeg")
    if found:
        return found
    candidate = Path("/home/yi.yang/miniforge3/envs/videogen-v2/bin/ffmpeg")
    if candidate.is_file():
        return str(candidate)
    raise SystemExit("ffmpeg not found")


def safe_title(info, cache_dir):
    title = str(info.get("title") or cache_dir.name).strip()
    title = re.sub(r'[\\/:*?"<>|\n\r]+', " ", title)
    title = re.sub(r"\s+", " ", title).strip(" .")
    return title or cache_dir.name


def fragment_kind(cache_dir, name):
    playurl = cache_dir / ".playurl"
    if not playurl.is_file():
        return None
    payload = json.loads(playurl.read_text(encoding="utf-8"))
    dash = payload.get("data", {}).get("dash", {})
    for kind in ("video", "audio"):
        for item in dash.get(kind) or []:
            url = item.get("baseUrl") or item.get("base_url") or ""
            if Path(url.split("?", 1)[0]).name == name:
                return kind
    return None


def strip_fragment(src, dest):
    raw = src.read_bytes()
    if raw.startswith(PREFIX) and raw[len(PREFIX):].find(b"ftyp") < 16:
        raw = raw[len(PREFIX):]
    else:
        at = raw.find(b"ftyp")
        if at < 4:
            raise SystemExit("not a media fragment: %s" % src)
        raw = raw[at - 4:]
    dest.write_bytes(raw)


def restore_one(cache_dir, ffmpeg):
    info_path = cache_dir / "videoInfo.json"
    if not info_path.is_file():
        return None
    fragments = sorted(cache_dir.glob("*.m4s"))
    if len(fragments) < 2:
        raise SystemExit("need video and audio fragments in %s" % cache_dir)
    info = json.loads(info_path.read_text(encoding="utf-8"))
    out = cache_dir.parent / ("%s.mp4" % safe_title(info, cache_dir))
    if out.exists():
        print("skip %s" % out)
        return out
    video = audio = None
    for fragment in fragments:
        kind = fragment_kind(cache_dir, fragment.name)
        if kind == "video":
            video = fragment
        elif kind == "audio":
            audio = fragment
    if video is None or audio is None:
        ordered = sorted(fragments, key=lambda path: path.stat().st_size, reverse=True)
        video, audio = ordered[0], ordered[1]
    with tempfile.TemporaryDirectory(prefix="bili-restore-") as tmp:
        tmp = Path(tmp)
        strip_fragment(video, tmp / "video.m4s")
        strip_fragment(audio, tmp / "audio.m4s")
        subprocess.check_call([
            ffmpeg, "-y", "-v", "error",
            "-i", str(tmp / "video.m4s"),
            "-i", str(tmp / "audio.m4s"),
            "-c", "copy", "-movflags", "+faststart",
            str(out),
        ])
    print("wrote %s" % out)
    return out


def caches_under(path):
    path = path.resolve()
    if (path / "videoInfo.json").is_file():
        return [path]
    return sorted(item for item in path.iterdir() if (item / "videoInfo.json").is_file())


def main():
    parser = argparse.ArgumentParser(description="Restore Bilibili m4s caches to mp4.")
    parser.add_argument("path", type=Path, help="One cache directory, or a folder of them.")
    args = parser.parse_args()
    ffmpeg = ffmpeg_bin()
    caches = caches_under(args.path)
    if not caches:
        raise SystemExit("no videoInfo.json under %s" % args.path)
    for cache in caches:
        restore_one(cache, ffmpeg)


if __name__ == "__main__":
    main()
