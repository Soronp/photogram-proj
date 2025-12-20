#!/usr/bin/env python3
"""
input.py

MARK-2 Input Ingestion Stage (Modern)
-------------------------------------
- Copies images → images_processed/
- Copies videos → videos/
- Extracts frames deterministically
- Accepts logger for pipeline parity
"""

from pathlib import Path
import shutil
import subprocess

from utils.paths import ProjectPaths

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}


def extract_video_frames(video: Path, out_dir: Path, fps: int = 2, logger=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = out_dir / f"{video.stem}_frame_%06d.jpg"

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video),
        "-vf", f"fps={fps}",
        str(pattern),
    ]

    msg = f"[input] Extracting frames from {video.name} → {out_dir}"
    if logger:
        logger.info(msg)
    else:
        print(msg)

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    if proc.stdout.strip():
        if logger:
            logger.info(proc.stdout)
        else:
            print(proc.stdout)

    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg failed for {video.name}")


def run(project_root: Path, input_path: Path, force: bool, logger=None):
    project_root = project_root.resolve()
    input_path = input_path.resolve()

    paths = ProjectPaths(project_root)
    paths.ensure_all()

    if not input_path.exists() or not input_path.is_dir():
        raise RuntimeError(f"Invalid input directory: {input_path}")

    copied_images = 0
    copied_videos = 0

    msg = "[input] Starting input ingestion"
    if logger:
        logger.info(msg)
    else:
        print(msg)

    for item in sorted(input_path.iterdir()):
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

    msg = f"[input] Copied images: {copied_images}, videos: {copied_videos}"
    if logger:
        logger.info(msg)
    else:
        print(msg)

    # Extract frames from all videos in the videos folder
    for video in sorted(paths.videos.iterdir()):
        if video.suffix.lower() in VIDEO_EXTS:
            extract_video_frames(video, paths.images_processed, fps=2, logger=logger)

    total_images = len(list(paths.images_processed.glob("*.jpg")))
    if total_images == 0:
        raise RuntimeError("No images available after ingestion")

    msg = f"[input] Total images ready: {total_images}"
    if logger:
        logger.info(msg)
    else:
        print(msg)


# --- CLI entrypoint ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MARK-2 Input Ingestion")
    parser.add_argument("--project", required=True, help="Project root directory")
    parser.add_argument("--input", required=True, help="Input directory")
    parser.add_argument("--force", action="store_true", help="Force re-ingest")
    args = parser.parse_args()

    run(Path(args.project), Path(args.input), args.force)
