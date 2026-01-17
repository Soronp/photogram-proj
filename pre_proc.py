#!/usr/bin/env python3
"""
pre_proc.py

MARK-2 Image Preprocessing Stage
--------------------------------
- Applies EXIF orientation
- Converts to high-quality JPEG
- Letterbox pads to square canvas
- Deterministic renaming
- Runner-managed logger ONLY
"""

import shutil
from pathlib import Path
from PIL import Image, ImageOps
import numpy as np

from utils.paths import ProjectPaths
from utils.config import load_config

# --------------------------------------------------
# Constants
# --------------------------------------------------
JPEG_QUALITY = 98
VALID_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
MAX_SAFE_RESOLUTION = 4608
MIN_RESOLUTION = 2000
PAD_COLOR = (0, 0, 0)


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in VALID_EXTS


def determine_target_resolution(images: list[Path], logger) -> int:
    dims = []

    for p in images:
        try:
            with Image.open(p) as img:
                dims.append(max(img.size))
        except Exception:
            logger.warning(f"[preprocess] Unreadable image skipped: {p.name}")

    if not dims:
        raise RuntimeError("Failed to determine image resolutions")

    median_dim = int(np.median(dims))
    count = len(images)

    if count <= 150:
        target = median_dim
        logger.info("[preprocess] Small dataset — preserving resolution")
    else:
        target = int(median_dim * 0.75)
        logger.info("[preprocess] Large dataset — applying downscale")

    target = min(target, MAX_SAFE_RESOLUTION)
    target = max(target, MIN_RESOLUTION)

    logger.info(f"[preprocess] Target canvas: {target}x{target}")
    return target


def preprocess_image(src: Path, dst: Path, target: int, logger) -> bool:
    try:
        img = Image.open(src)
    except Exception as e:
        logger.warning(f"[preprocess] Skipped unreadable image: {src.name} ({e})")
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

    canvas.save(
        dst,
        format="JPEG",
        quality=JPEG_QUALITY,
        subsampling=0,
        optimize=True,
    )

    return True


# --------------------------------------------------
# Pipeline Stage
# --------------------------------------------------
def run(run_root: Path, project_root: Path, force: bool, logger):
    """
    MARK-2 pipeline stage: image preprocessing
    """
    paths = ProjectPaths(project_root)
    paths.ensure_all()
    load_config(run_root)  # reserved for future use

    logger.info("[preprocess] Stage started")

    src = paths.images_filtered
    dst = paths.images_processed

    if not src.exists():
        raise RuntimeError("images_filtered/ missing")

    if dst.exists() and any(dst.iterdir()):
        if not force:
            logger.info("[preprocess] Output exists — skipping")
            return
        logger.info("[preprocess] Force enabled — clearing images_processed")
        shutil.rmtree(dst)

    dst.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in src.iterdir() if is_image_file(p))
    if not images:
        raise RuntimeError("No valid images found for preprocessing")

    target = determine_target_resolution(images, logger)

    processed = 0
    for i, img in enumerate(images):
        out = dst / f"img_{i:06d}.jpg"
        if preprocess_image(img, out, target, logger):
            processed += 1

    if processed == 0:
        raise RuntimeError("Preprocessing produced zero images")

    logger.info(f"[preprocess] Processed images: {processed}")
    logger.info("[preprocess] Stage completed successfully")
