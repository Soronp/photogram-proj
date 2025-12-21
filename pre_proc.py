#!/usr/bin/env python3
"""
pre_processing.py

MARK-2 Image Preprocessing Stage (OpenMVS-Accurate)
--------------------------------------------------
- Converts filtered images to COLMAP-friendly JPEG
- Applies EXIF orientation
- Preserves high-resolution geometry
- Letterbox pads to uniform square canvas
- Sequentially renames images
"""

import shutil
from pathlib import Path
from PIL import Image, ImageOps
import numpy as np
import logging

from utils.paths import ProjectPaths
from utils.config import load_config

# -----------------------------
# Constants
# -----------------------------
JPEG_QUALITY = 98
VALID_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
MAX_SAFE_RESOLUTION = 4608
MIN_RESOLUTION = 2000
PAD_COLOR = (0, 0, 0)

# -----------------------------
# Logger
# -----------------------------
def make_logger(name: str, log_dir: Path):
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        fh = logging.FileHandler(log_dir / f"{name}.log")
        sh = logging.StreamHandler()
        fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
        fh.setFormatter(fmt)
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)

    return logger

# -----------------------------
# Helpers
# -----------------------------
def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in VALID_EXTS

def determine_target_resolution(images: list[Path], logger) -> int:
    max_dims = []
    for p in images:
        try:
            with Image.open(p) as img:
                max_dims.append(max(img.size))
        except Exception:
            logger.warning(f"Unreadable image skipped: {p.name}")

    if not max_dims:
        raise RuntimeError("Failed to determine image resolutions")

    median_dim = int(np.median(max_dims))
    image_count = len(images)

    if image_count <= 150:
        target = median_dim
        logger.info("Small dataset — preserving native resolution")
    else:
        target = int(median_dim * 0.75)
        logger.info("Large dataset — moderate downscaling")

    target = min(target, MAX_SAFE_RESOLUTION)
    target = max(target, MIN_RESOLUTION)
    logger.info(f"Target canvas size: {target}x{target}px")
    return target

def preprocess_image(src: Path, dst: Path, target: int, logger) -> bool:
    try:
        img = Image.open(src)
    except Exception as e:
        logger.warning(f"Unreadable image skipped: {src.name} ({e})")
        return False

    img = ImageOps.exif_transpose(img)

    if img.mode != "RGB":
        img = img.convert("RGB")

    w, h = img.size
    scale = target / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)

    img = img.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGB", (target, target), PAD_COLOR)
    canvas.paste(img, ((target - new_w) // 2, (target - new_h) // 2))

    canvas.save(dst, format="JPEG", quality=JPEG_QUALITY, subsampling=0, optimize=True)
    return True

# -----------------------------
# Pipeline
# -----------------------------
def run(run_root: Path, project_root: Path, force: bool, logger):
    """
    Modernized pre-processing stage to work with runner:
    Accepts (project_root, force, logger)
    """
    paths = ProjectPaths(project_root)
    paths.ensure_all()
    load_config(project_root)

    if logger is None:
        logger = make_logger("pre_processing", paths.logs)

    src = paths.images_filtered
    dst = paths.images_processed

    if not src.exists():
        raise RuntimeError("images_filtered/ missing")

    if dst.exists() and any(dst.iterdir()):
        if not force:
            logger.info("images_processed exists — skipping")
            return
        logger.warning("Force enabled — clearing images_processed")
        shutil.rmtree(dst)

    dst.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in src.iterdir() if is_image_file(p))
    if not images:
        raise RuntimeError("No valid images in images_filtered")

    target = determine_target_resolution(images, logger)

    processed = 0
    for i, img in enumerate(images):
        out = dst / f"img_{i:06d}.jpg"
        if preprocess_image(img, out, target, logger):
            processed += 1

    if processed == 0:
        raise RuntimeError("No images were successfully processed")

    logger.info(f"Preprocessing complete: {processed} images")
    logger.info(f"Final resolution: {target}x{target}")

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run(Path(args.project), args.force)
