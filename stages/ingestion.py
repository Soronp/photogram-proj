#!/usr/bin/env python3
"""
Stage 1 — Dataset ingestion
"""

import shutil
from pathlib import Path


# --------------------------------------------------
# Video frame extraction
# --------------------------------------------------

def extract_frames(video: Path, output_dir: Path, tools, logger):

    logger.info(f"[ingestion] extracting frames from {video.name}")

    pattern = output_dir / f"{video.stem}_%05d.jpg"

    cmd = [
        "-i", str(video),
        "-qscale:v", "2",
        str(pattern)
    ]

    tools.run("ffmpeg", cmd)


# --------------------------------------------------
# Stage
# --------------------------------------------------

def run(paths, logger, tools, config):

    logger.info("[ingestion] starting")

    dataset = Path(config["project"]["dataset_path"]).resolve()

    if not dataset.exists():
        raise FileNotFoundError(dataset)

    paths.ensure()

    image_ext = {".jpg", ".jpeg", ".png"}
    video_ext = {".mp4", ".mov", ".avi", ".mkv"}

    images = []
    videos = []

    for f in sorted(dataset.iterdir()):

        if f.suffix.lower() in image_ext:
            images.append(f)

        elif f.suffix.lower() in video_ext:
            videos.append(f)

    # --------------------------------------------------
    # Copy images
    # --------------------------------------------------

    for i, src in enumerate(images):

        dst = paths.images / f"img_{i:05d}{src.suffix.lower()}"

        shutil.copy2(src, dst)

    if images:
        logger.info(f"[ingestion] imported {len(images)} images")

    # --------------------------------------------------
    # Process videos
    # --------------------------------------------------

    for video in videos:
        extract_frames(video, paths.images, tools, logger)

    if videos:
        logger.info(f"[ingestion] processed {len(videos)} videos")

    total = len(list(paths.images.glob("*")))

    if total == 0:
        raise RuntimeError("No images produced")

    logger.info(f"[ingestion] total images: {total}")

    logger.info("[ingestion] completed")