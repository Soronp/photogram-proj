#!/usr/bin/env python3
"""
input_ingest.py

Normalize raw inputs into a clean image dataset.

Responsibilities:
- Detect input type (images vs video)
- Copy or extract frames into images/
- Prevent duplicate frames/images
- Enforce normalized frame naming
- Record frame metadata

Assumptions:
- ffmpeg is available in PATH
- This script is safe to re-run
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


def parse_args():
    parser = argparse.ArgumentParser(description="Ingest images or video into project")
    parser.add_argument("--project", required=True, help="Path to project root")
    parser.add_argument("--force", action="store_true", help="Overwrite existing images")
    return parser.parse_args()


def file_hash(path: Path, chunk_size=8192) -> str:
    """Compute SHA1 hash for duplicate detection."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def ingest_images(raw_dir: Path, images_dir: Path, logger):
    logger.info("Ingesting image files")

    seen_hashes = {}
    frame_index = 0
    metadata = []

    for src in sorted(raw_dir.iterdir()):
        if src.suffix.lower() not in SUPPORTED_IMAGE_EXTS:
            continue

        h = file_hash(src)
        if h in seen_hashes:
            logger.warning(f"Duplicate image skipped: {src.name}")
            continue

        dst_name = f"frame_{frame_index:06d}.jpg"
        dst = images_dir / dst_name

        shutil.copy2(src, dst)
        seen_hashes[h] = dst_name

        metadata.append({
            "frame": dst_name,
            "source": src.name,
            "hash": h,
        })

        frame_index += 1

    return metadata


def ingest_video(raw_dir: Path, images_dir: Path, logger):
    logger.info("Ingesting video files")

    temp_dir = images_dir / "_ffmpeg_tmp"
    temp_dir.mkdir(exist_ok=True)

    metadata = []
    frame_index = 0
    seen_hashes = {}

    for video in sorted(raw_dir.iterdir()):
        if video.suffix.lower() not in SUPPORTED_VIDEO_EXTS:
            continue

        logger.info(f"Extracting frames from {video.name}")

        cmd = [
            "ffmpeg",
            "-i", str(video),
            "-vsync", "vfr",
            "-q:v", "2",
            str(temp_dir / "%06d.jpg"),
        ]

        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        for frame in sorted(temp_dir.iterdir()):
            h = file_hash(frame)
            if h in seen_hashes:
                logger.debug(f"Duplicate frame skipped: {frame.name}")
                frame.unlink()
                continue

            dst_name = f"frame_{frame_index:06d}.jpg"
            dst = images_dir / dst_name
            shutil.move(frame, dst)

            seen_hashes[h] = dst_name
            metadata.append({
                "frame": dst_name,
                "source": video.name,
                "hash": h,
            })

            frame_index += 1

        shutil.rmtree(temp_dir)
        temp_dir.mkdir(exist_ok=True)

    temp_dir.rmdir()
    return metadata


def main():
    args = parse_args()

    project_root = Path(args.project).resolve()
    paths = ProjectPaths(project_root)
    paths.ensure_all()

    logger = get_logger("ingest", project_root)
    logger.info("Starting input ingestion")

    raw_dir = paths.raw
    images_dir = paths.images

    if images_dir.exists() and any(images_dir.iterdir()) and not args.force:
        logger.warning("images/ already populated; use --force to re-ingest")
        return

    # Clear images dir if force
    if args.force and images_dir.exists():
        shutil.rmtree(images_dir)
        images_dir.mkdir()

    raw_files = list(raw_dir.iterdir())
    if not raw_files:
        logger.error("No input files found in raw/")
        return

    image_files = [f for f in raw_files if f.suffix.lower() in SUPPORTED_IMAGE_EXTS]
    video_files = [f for f in raw_files if f.suffix.lower() in SUPPORTED_VIDEO_EXTS]

    if image_files and video_files:
        logger.error("Mixed image and video inputs are not supported")
        return

    if image_files:
        metadata = ingest_images(raw_dir, images_dir, logger)
        input_type = "images"
    else:
        metadata = ingest_video(raw_dir, images_dir, logger)
        input_type = "video"

    frames_json = paths.images / "frames.json"
    with open(frames_json, "w", encoding="utf-8") as f:
        json.dump({
            "input_type": input_type,
            "frame_count": len(metadata),
            "frames": metadata,
        }, f, indent=2)

    logger.info(f"Ingested {len(metadata)} frames")
    logger.info("Input ingestion complete")


if __name__ == "__main__":
    main()
