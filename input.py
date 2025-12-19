#!/usr/bin/env python3
"""
input.py

MARK-2 Input Ingestion Stage
---------------------------
- Accepts image folders and/or video files
- Copies images directly into images_processed/
- Copies videos into videos/
- Extracts video frames using FFmpeg
- Outputs a unified image set into images_processed/
- Deterministic, logged, restart-safe
"""

from pathlib import Path
import shutil
import subprocess

from utils.logger import get_logger
from utils.paths import ProjectPaths

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}


# ------------------------------------------------------------------
# Video → frame extraction
# ------------------------------------------------------------------
def extract_video_frames(
    video_path: Path,
    output_dir: Path,
    logger,
    fps: int = 2,
):
    """
    Deterministically extract frames from a video using FFmpeg.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_pattern = output_dir / f"{video_path.stem}_frame_%06d.jpg"

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vf", f"fps={fps}",
        str(frame_pattern),
    ]

    logger.info(f"Extracting frames from video: {video_path.name}")
    logger.info(f"FFmpeg command: {' '.join(cmd)}")

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    logger.info(proc.stdout)

    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg failed for video: {video_path.name}")


# ------------------------------------------------------------------
# Input stage
# ------------------------------------------------------------------
def run(project_root: Path, input_path: Path, force: bool = False):
    project_root = project_root.resolve()
    input_path = input_path.resolve()

    paths = ProjectPaths(project_root)
    paths.ensure_all()

    logger = get_logger("input", project_root)

    logger.info("=== MARK-2 Input Ingestion ===")
    logger.info(f"Input source : {input_path}")
    logger.info(f"Project root : {project_root}")

    if not input_path.exists() or not input_path.is_dir():
        raise RuntimeError(f"Invalid input folder: {input_path}")

    # Ensure videos directory exists (TOP-LEVEL, not raw/)
    videos_dir = paths.root / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    copied_images = 0
    copied_videos = 0

    # --------------------------------------------------
    # Copy inputs
    # --------------------------------------------------
    for item in sorted(input_path.iterdir()):
        suffix = item.suffix.lower()

        # -----------------------------
        # Images → images_processed
        # -----------------------------
        if suffix in IMAGE_EXTS:
            dest = paths.images_processed / item.name
            if dest.exists() and not force:
                logger.warning(f"Skipping existing image: {item.name}")
                continue
            shutil.copy2(item, dest)
            copied_images += 1

        # -----------------------------
        # Videos → videos/
        # -----------------------------
        elif suffix in VIDEO_EXTS:
            dest = videos_dir / item.name
            if dest.exists() and not force:
                logger.warning(f"Skipping existing video: {item.name}")
                continue
            shutil.copy2(item, dest)
            copied_videos += 1

    logger.info(f"Copied {copied_images} images")
    logger.info(f"Copied {copied_videos} videos")

    # --------------------------------------------------
    # Extract frames from videos
    # --------------------------------------------------
    for video in sorted(videos_dir.iterdir()):
        if video.suffix.lower() in VIDEO_EXTS:
            extract_video_frames(
                video_path=video,
                output_dir=paths.images_processed,
                logger=logger,
                fps=2,  # deterministic default
            )

    total_images = len(list(paths.images_processed.glob("*.jpg")))
    if total_images == 0:
        raise RuntimeError("No images available after input ingestion")

    logger.info(f"Total images ready for pipeline: {total_images}")
    logger.info("Input ingestion completed successfully")
