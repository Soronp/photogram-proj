from pathlib import Path
from PIL import Image


def run(paths, config, logger):
    stage = "validate_images"
    logger.info(f"---- {stage.upper()} ----")

    img_dir = paths.images

    if not img_dir.exists():
        raise RuntimeError(f"{stage}: images directory not found: {img_dir}")

    valid_images = []
    invalid_images = []

    widths = []
    heights = []

    for img_path in img_dir.iterdir():
        try:
            with Image.open(img_path) as img:
                img.verify()

            # Reopen to get size (verify() closes file)
            with Image.open(img_path) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)

            valid_images.append(img_path)

        except Exception:
            logger.warning(f"{stage}: corrupted image -> {img_path.name}")
            invalid_images.append(img_path)

    # -----------------------------
    # Remove corrupted images
    # -----------------------------
    for img in invalid_images:
        try:
            img.unlink()
        except Exception:
            logger.warning(f"{stage}: failed to delete {img.name}")

    # -----------------------------
    # Stats
    # -----------------------------
    num_valid = len(valid_images)
    num_invalid = len(invalid_images)

    logger.info(f"{stage}: valid={num_valid}, invalid_removed={num_invalid}")

    if num_valid == 0:
        raise RuntimeError(f"{stage}: No valid images left after validation")

    avg_w = sum(widths) // len(widths)
    avg_h = sum(heights) // len(heights)

    logger.info(f"{stage}: avg_resolution={avg_w}x{avg_h}")
    logger.info(f"{stage}: DONE")