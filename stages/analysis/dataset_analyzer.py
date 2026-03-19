from pathlib import Path
import cv2
import numpy as np


def compute_entropy(gray_img):
    """
    Compute Shannon entropy of grayscale image.
    """
    hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
    hist = hist.ravel() / hist.sum()

    hist = hist[hist > 0]  # remove zeros

    return -np.sum(hist * np.log2(hist))


def run(paths, config, logger):
    stage = "dataset_analyzer"
    logger.info(f"---- {stage.upper()} ----")

    # -----------------------------
    # Select image directory
    # -----------------------------
    downsample_enabled = config.get("downsampling", {}).get("enabled", True)
    image_dir = paths.images_downsampled if downsample_enabled else paths.images

    if not image_dir.exists():
        raise RuntimeError(f"{stage}: image directory not found")

    image_files = list(image_dir.glob("*"))

    if len(image_files) == 0:
        raise RuntimeError(f"{stage}: no images found")

    logger.info(f"{stage}: analyzing {len(image_files)} images")

    # -----------------------------
    # Stats containers
    # -----------------------------
    widths = []
    heights = []
    entropies = []

    # -----------------------------
    # Process images
    # -----------------------------
    for img_path in image_files:
        try:
            img = cv2.imread(str(img_path))

            if img is None:
                logger.warning(f"{stage}: failed to read {img_path.name}")
                continue

            h, w = img.shape[:2]
            widths.append(w)
            heights.append(h)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ent = compute_entropy(gray)
            entropies.append(ent)

        except Exception as e:
            logger.warning(f"{stage}: error processing {img_path.name}: {e}")

    # -----------------------------
    # Aggregate stats
    # -----------------------------
    stats = {
        "num_images": len(widths),
        "avg_width": float(np.mean(widths)),
        "avg_height": float(np.mean(heights)),
        "max_dim": int(max(max(widths), max(heights))),
        "avg_entropy": float(np.mean(entropies)),
        "min_entropy": float(np.min(entropies)),
        "max_entropy": float(np.max(entropies)),
    }

    # -----------------------------
    # Derived metrics
    # -----------------------------
    stats["aspect_variation"] = float(np.std(
        [w / h for w, h in zip(widths, heights)]
    ))

    logger.info(f"{stage}: stats computed")
    logger.info(f"{stage}: {stats}")

    return stats