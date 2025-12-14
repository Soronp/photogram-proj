#!/usr/bin/env python3
"""
preprocess_images.py

MARK-2 Image Preprocessing Stage
--------------------------------
- Converts images to COLMAP-friendly JPEG
- Applies EXIF orientation
- Downsamples to a fixed max resolution
- Renames images sequentially: img_000000.jpg, img_000001.jpg, ...
- Produces canonical images_processed/ directory
- Must run before sparse reconstruction
"""

import argparse
from pathlib import Path
from PIL import Image, ImageOps

from utils.logger import get_logger
from utils.paths import ProjectPaths
from utils.config import load_config

# --------------------------------------------------
# Configuration defaults
# --------------------------------------------------

DEFAULT_MAX_SIZE = 2000
JPEG_QUALITY = 95

# --------------------------------------------------
# Helpers
# --------------------------------------------------

def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

def preprocess_image(
    src: Path,
    dst: Path,
    max_size: int,
    logger
):
    """Preprocess a single image and save as a COLMAP-friendly JPEG."""
    try:
        img = Image.open(src)
    except Exception as e:
        logger.warning(f"Skipping unreadable image: {src.name} ({e})")
        return False

    # Apply EXIF orientation
    img = ImageOps.exif_transpose(img)

    # Convert to RGB
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Downsample if necessary
    w, h = img.size
    scale = min(1.0, max_size / max(w, h))
    if scale < 1.0:
        new_size = (int(w * scale), int(h * scale))
        img = img.resize(new_size, Image.LANCZOS)
        logger.debug(f"{src.name}: resized {w}x{h} → {new_size[0]}x{new_size[1]}")

    # Save to destination
    img.save(dst, format="JPEG", quality=JPEG_QUALITY, subsampling=0, optimize=True)
    return True

# --------------------------------------------------
# Main preprocessing logic
# --------------------------------------------------

def run_preprocessing(project_root: Path, force: bool):
    paths = ProjectPaths(project_root)
    config = load_config(project_root)
    logger = get_logger("preprocess_images", project_root)

    raw_dir = paths.raw
    out_dir = paths.images_processed
    max_size = int(config.get("preprocess_max_image_size", DEFAULT_MAX_SIZE))

    logger.info("Starting image preprocessing stage")
    logger.info(f"Raw images: {raw_dir}")
    logger.info(f"Output images: {out_dir}")
    logger.info(f"Max image size: {max_size}px")

    if not raw_dir.exists():
        raise FileNotFoundError("raw/ directory does not exist")

    if out_dir.exists():
        if force:
            logger.warning("Removing existing images_processed (--force)")
            for f in out_dir.iterdir():
                f.unlink()
        else:
            logger.info("images_processed already exists — skipping")
            return

    out_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in raw_dir.iterdir() if is_image_file(p))
    if not images:
        raise RuntimeError("No valid image files found in raw/")

    processed = 0
    for i, img_path in enumerate(images):
        dst_name = f"img_{i:06d}.jpg"
        dst_path = out_dir / dst_name
        ok = preprocess_image(
            src=img_path,
            dst=dst_path,
            max_size=max_size,
            logger=logger
        )
        if ok:
            processed += 1

    if processed == 0:
        raise RuntimeError("No images were successfully processed")

    logger.info(f"Preprocessing complete: {processed} images written")
    logger.info("Canonical image naming and resolution frozen for pipeline")

# --------------------------------------------------
# CLI
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MARK-2 Image Preprocessing")
    parser.add_argument("project_root", type=Path)
    parser.add_argument("--force", action="store_true", help="Overwrite existing processed images")
    args = parser.parse_args()
    run_preprocessing(args.project_root, args.force)

if __name__ == "__main__":
    main()
