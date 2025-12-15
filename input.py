#!/usr/bin/env python3
"""
Normalize raw inputs into a clean image dataset.
"""

import argparse
import subprocess
import hashlib
import json
import shutil
from pathlib import Path

from utils.logger import get_logger
from utils.paths import ProjectPaths

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
SUPPORTED_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def file_hash(path: Path, chunk_size=8192) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def ingest_images(raw_dir, images_dir, logger):
    seen = {}
    metadata = []
    idx = 0

    for src in sorted(raw_dir.iterdir()):
        if src.suffix.lower() not in SUPPORTED_IMAGE_EXTS:
            continue

        h = file_hash(src)
        if h in seen:
            logger.warning(f"Duplicate image skipped: {src.name}")
            continue

        dst = images_dir / f"frame_{idx:06d}.jpg"
        shutil.copy2(src, dst)

        metadata.append({
            "frame": dst.name,
            "source": src.name,
            "hash": h,
        })

        seen[h] = dst.name
        idx += 1

    return metadata


def ingest_video(raw_dir, images_dir, logger):
    temp = images_dir / "_ffmpeg_tmp"
    temp.mkdir(exist_ok=True)

    seen = {}
    metadata = []
    idx = 0

    for video in sorted(raw_dir.iterdir()):
        if video.suffix.lower() not in SUPPORTED_VIDEO_EXTS:
            continue

        logger.info(f"Extracting frames from {video.name}")

        cmd = [
            "ffmpeg", "-i", str(video),
            "-vsync", "vfr",
            "-q:v", "2",
            str(temp / "%06d.jpg"),
        ]

        subprocess.run(cmd, check=True,
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)

        for frame in sorted(temp.iterdir()):
            h = file_hash(frame)
            if h in seen:
                frame.unlink()
                continue

            dst = images_dir / f"frame_{idx:06d}.jpg"
            shutil.move(frame, dst)

            metadata.append({
                "frame": dst.name,
                "source": video.name,
                "hash": h,
            })

            seen[h] = dst.name
            idx += 1

        shutil.rmtree(temp)
        temp.mkdir()

    temp.rmdir()
    return metadata


# --------------------------------------------------
# CORE CALLABLE (RUNNER USES THIS)
# --------------------------------------------------

def run(project_root: Path, force: bool = False):
    paths = ProjectPaths(project_root)
    paths.ensure_all()

    logger = get_logger("ingest", project_root)
    logger.info("Starting input ingestion")

    raw_dir = paths.raw
    images_dir = paths.images

    if images_dir.exists() and any(images_dir.iterdir()) and not force:
        logger.warning("images/ already populated; skipping")
        return

    if force and images_dir.exists():
        shutil.rmtree(images_dir)
        images_dir.mkdir()

    raw_files = list(raw_dir.iterdir())
    if not raw_files:
        raise RuntimeError("No input files found in raw/")

    images = [f for f in raw_files if f.suffix.lower() in SUPPORTED_IMAGE_EXTS]
    videos = [f for f in raw_files if f.suffix.lower() in SUPPORTED_VIDEO_EXTS]

    if images and videos:
        raise RuntimeError("Mixed image and video inputs not supported")

    if images:
        metadata = ingest_images(raw_dir, images_dir, logger)
        input_type = "images"
    else:
        metadata = ingest_video(raw_dir, images_dir, logger)
        input_type = "video"

    with open(images_dir / "frames.json", "w", encoding="utf-8") as f:
        json.dump({
            "input_type": input_type,
            "frame_count": len(metadata),
            "frames": metadata,
        }, f, indent=2)

    logger.info(f"Ingested {len(metadata)} frames")


# --------------------------------------------------
# CLI WRAPPER
# --------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--project", required=True)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    run(Path(args.project), args.force)


if __name__ == "__main__":
    main()
