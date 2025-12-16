#!/usr/bin/env python3
"""
pre_processing.py

MARK-2 Image Preprocessing Stage
--------------------------------
- Converts filtered images to COLMAP-friendly JPEG
- Applies EXIF orientation
- Downsamples to max resolution defined in config
- Sequentially renames images: img_000000.jpg, img_000001.jpg, ...
- Produces canonical images_processed/ directory

Stage order:
    input -> image_filter -> pre_processing -> database_builder
"""

import shutil
from pathlib import Path
from PIL import Image, ImageOps

from utils.logger import get_logger
from utils.paths import ProjectPaths
from utils.config import load_config

# -----------------------------
# Defaults
# -----------------------------
DEFAULT_MAX_SIZE = 2000
JPEG_QUALITY = 95
VALID_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

# -----------------------------
# Helpers
# -----------------------------
def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in VALID_EXTS

def preprocess_image(src: Path, dst: Path, max_size: int, logger) -> tuple[bool, tuple[int, int]|None]:
    """Preprocess single image. Returns (success, final_dimensions)."""
    try:
        img = Image.open(src)
    except Exception as e:
        logger.warning(f"Unreadable image skipped: {src.name} ({e})")
        return False, None

    # Apply EXIF orientation
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        logger.debug(f"No EXIF or error in {src.name}, using as-is")

    if img.mode != "RGB":
        img = img.convert("RGB")

    original_w, original_h = img.size
    scale = min(1.0, max_size / max(original_w, original_h))

    if scale < 1.0:
        new_w = int(original_w * scale)
        new_h = int(original_h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        final_dims = (new_w, new_h)
        logger.debug(f"{src.name}: {original_w}x{original_h} → {new_w}x{new_h}")
    else:
        final_dims = (original_w, original_h)
        logger.debug(f"{src.name}: unchanged {original_w}x{original_h}")

    img.save(
        dst,
        format="JPEG",
        quality=JPEG_QUALITY,
        subsampling=0,  # 4:4:4
        optimize=True,
    )
    return True, final_dims

# -----------------------------
# Pipeline entrypoint
# -----------------------------
def run(project_root: Path, force: bool):
    paths = ProjectPaths(project_root)
    paths.ensure_all()

    logger = get_logger("pre_processing", project_root)
    config = load_config(project_root)

    src_dir = paths.images_filtered
    out_dir = paths.images_processed
    max_size = int(config.get("preprocess_max_image_size", DEFAULT_MAX_SIZE))

    logger.info("Starting image preprocessing")
    logger.info(f"Filtered images: {src_dir}")
    logger.info(f"Processed images output: {out_dir}")
    logger.info(f"Max image size: {max_size}px")

    if not src_dir.exists():
        raise RuntimeError(f"Input directory does not exist: {src_dir}")

    # Skip / force logic
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

    processed = 0
    dimension_counts = {}

    for idx, img_path in enumerate(images):
        dst_name = f"img_{idx:06d}.jpg"
        dst_path = out_dir / dst_name

        success, final_dims = preprocess_image(
            src=img_path,
            dst=dst_path,
            max_size=max_size,
            logger=logger,
        )
        if success:
            processed += 1
            if final_dims:
                dimension_counts[final_dims] = dimension_counts.get(final_dims, 0) + 1

    if processed == 0:
        raise RuntimeError("No images were successfully preprocessed")

    # Dimension statistics
    logger.info(f"Preprocessing complete: {processed} images written")
    if len(dimension_counts) == 1:
        dim = list(dimension_counts.keys())[0]
        logger.info(f"All images: {dim[0]}x{dim[1]} pixels")
    else:
        logger.warning("Multiple image dimensions detected:")
        for dim, count in sorted(dimension_counts.items()):
            logger.warning(f"  {dim[0]}x{dim[1]}: {count} images")
        logger.warning("Different dimensions may affect reconstruction downstream")

    logger.info("Preprocessed images ready for database building and matching")
