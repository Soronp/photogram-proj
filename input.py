#!/usr/bin/env python3
"""
input.py

MARK-2 Input Ingestion Stage
---------------------------
- Copy images into images_processed/
- Copy videos into videos/
- Deterministic frame extraction
"""

from pathlib import Path
import shutil
import subprocess

from utils.paths import ProjectPaths

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}


def extract_video_frames(video: Path, out_dir: Path, fps: int, logger):
    out_dir.mkdir(parents=True, exist_ok=True)

    pattern = out_dir / f"{video.stem}_frame_%06d.jpg"
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video),
        "-vf", f"fps={fps}",
        str(pattern),
    ]

    logger.info(f"[input] Extracting frames from {video.name}")

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    if proc.stdout.strip():
        logger.debug(proc.stdout)

    if proc.returncode != 0:
        raise RuntimeError(f"[input] FFmpeg failed for {video.name}")


def run(run_root: Path, project_root: Path, force: bool, logger, *, input_path: Path):
    project_root = project_root.resolve()
    input_path = input_path.resolve()

    paths = ProjectPaths(project_root)
    paths.ensure_all()

    if not input_path.is_dir():
        raise RuntimeError(f"[input] Invalid input directory: {input_path}")

    logger.info("[input] Starting ingestion")

    copied_images = 0
    copied_videos = 0

    for item in sorted(input_path.iterdir()):
        if not item.is_file():
            continue

        ext = item.suffix.lower()

        if ext in IMAGE_EXTS:
            dst = paths.images_processed / item.name
            if dst.exists() and not force:
                continue
            shutil.copy2(item, dst)
            copied_images += 1

        elif ext in VIDEO_EXTS:
            dst = paths.videos / item.name
            if dst.exists() and not force:
                continue
            shutil.copy2(item, dst)
            copied_videos += 1

    logger.info(f"[input] Copied {copied_images} images, {copied_videos} videos")

    for video in sorted(paths.videos.iterdir()):
        if video.suffix.lower() in VIDEO_EXTS:
            extract_video_frames(video, paths.images_processed, fps=2, logger=logger)

    total_images = len(list(paths.images_processed.glob("*.jpg")))
    if total_images == 0:
        raise RuntimeError("[input] No images available after ingestion")

    logger.info(f"[input] Total images ready: {total_images}")
    logger.info("[input] Stage completed")
