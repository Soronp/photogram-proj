#!/usr/bin/env python3
"""
pre_processing.py

MARK-2 Image Preprocessing Stage (OpenMVS-Accurate)
--------------------------------------------------
- Converts filtered images to COLMAP-friendly JPEG
- Applies EXIF orientation
- Preserves high-resolution geometry for OpenMVS
- Enforces uniform image dimensions (LETTERBOX PAD)
- Sequentially renames images
"""

import shutil
from pathlib import Path
from PIL import Image, ImageOps
import numpy as np

from utils.logger import get_logger
from utils.paths import ProjectPaths
from utils.config import load_config

# -----------------------------
# Constants
# -----------------------------
JPEG_QUALITY = 98
VALID_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

MAX_SAFE_RESOLUTION = 4608
MIN_RESOLUTION = 2000
PAD_COLOR = (0, 0, 0)  # black padding (geometry-safe)

# -----------------------------
# Helpers
# -----------------------------
def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in VALID_EXTS


def determine_target_resolution(images: list[Path], logger) -> int:
    max_dims = []
    for p in images:
        try:
            with Image.open(p) as img:
                max_dims.append(max(img.size))
        except Exception:
            logger.debug(f"Unreadable image skipped: {p.name}")

    if not max_dims:
        raise RuntimeError("Failed to determine image resolutions")

    median_dim = int(np.median(max_dims))
    image_count = len(images)

    if image_count <= 150:
        target = median_dim
        logger.info("Small dataset detected — preserving native resolution")
    else:
        target = int(median_dim * 0.75)
        logger.info("Large dataset detected — moderate downscaling")

    target = min(target, MAX_SAFE_RESOLUTION)
    target = max(target, MIN_RESOLUTION)

    logger.info(
        f"Target canvas size: {target}x{target}px "
        f"(median={median_dim}px, images={image_count})"
    )
    return target


def preprocess_image(
    src: Path,
    dst: Path,
    target_size: int,
    logger,
):
    try:
        img = Image.open(src)
    except Exception as e:
        logger.warning(f"Unreadable image skipped: {src.name} ({e})")
        return False

    # EXIF orientation
    img = ImageOps.exif_transpose(img)

    # RGB enforcement
    if img.mode != "RGB":
        img = img.convert("RGB")

    w, h = img.size
    scale = target_size / max(w, h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    img = img.resize((new_w, new_h), Image.LANCZOS)

    # Letterbox pad to exact canvas
    canvas = Image.new("RGB", (target_size, target_size), PAD_COLOR)
    offset_x = (target_size - new_w) // 2
    offset_y = (target_size - new_h) // 2
    canvas.paste(img, (offset_x, offset_y))

    canvas.save(
        dst,
        format="JPEG",
        quality=JPEG_QUALITY,
        subsampling=0,
        optimize=True,
    )

    logger.debug(
        f"{src.name}: {w}x{h} → {new_w}x{new_h} padded to {target_size}x{target_size}"
    )
    return True


# -----------------------------
# Pipeline entrypoint
# -----------------------------
def run(project_root: Path, force: bool):
    paths = ProjectPaths(project_root)
    paths.ensure_all()

    logger = get_logger("pre_processing", project_root)
    load_config(project_root)

    src_dir = paths.images_filtered
    out_dir = paths.images_processed

    if not src_dir.exists():
        raise RuntimeError(f"Input directory does not exist: {src_dir}")

    if out_dir.exists() and any(out_dir.iterdir()):
        if not force:
            logger.info("images_processed exists; skipping preprocessing")
            return
        logger.warning("Force enabled — removing existing images_processed")
        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in src_dir.iterdir() if is_image_file(p))
    if not images:
        raise RuntimeError("No valid images found in images_filtered/")

    target_size = determine_target_resolution(images, logger)

    processed = 0
    for idx, img_path in enumerate(images):
        dst = out_dir / f"img_{idx:06d}.jpg"
        if preprocess_image(img_path, dst, target_size, logger):
            processed += 1

    if processed == 0:
        raise RuntimeError("No images were successfully preprocessed")

    logger.info(f"Preprocessing complete: {processed} images")
    logger.info(f"All images standardized to: {target_size}x{target_size}")
    logger.info("Preprocessed images ready for database building and matching")
