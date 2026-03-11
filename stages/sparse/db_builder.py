#!/usr/bin/env python3
"""
db_builder.py

Stage 4 — COLMAP Feature Extraction

Responsibilities
----------------
• collect filtered images
• create COLMAP database
• run COLMAP feature_extractor
• validate produced database
"""

import sqlite3
from pathlib import Path
from PIL import Image


VALID_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def collect_images(images_dir: Path):
    """
    Collect valid images from the filtered dataset.
    """

    return sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXTS
    )


def compute_max_image_size(images):
    """
    Determine safe max image size for SIFT extraction.
    """

    dims = []

    for p in images:
        try:
            with Image.open(p) as img:
                dims.append(max(img.size))
        except Exception:
            continue

    if not dims:
        raise RuntimeError("Could not determine image dimensions")

    return min(max(dims), 3072)


def auto_max_features(n_images):
    """
    Adaptive feature count depending on dataset size.
    """

    if n_images < 80:
        return 20000
    if n_images < 150:
        return 18000
    if n_images < 300:
        return 15000
    if n_images < 800:
        return 12000

    return 9000


# --------------------------------------------------
# Stage execution
# --------------------------------------------------

def run(paths, logger, tools, config):

    logger.info("[database] stage started")

    images_dir = paths.images_filtered
    db_path = paths.database_path

    if not images_dir.exists():
        raise RuntimeError("images_filtered directory missing")

    paths.database.mkdir(parents=True, exist_ok=True)

    if db_path.exists():
        logger.info("[database] removing previous database")
        db_path.unlink()

    images = collect_images(images_dir)

    if not images:
        raise RuntimeError("No images available for feature extraction")

    max_image = compute_max_image_size(images)
    max_features = auto_max_features(len(images))

    cam_cfg = config.get("camera", {})
    feat_cfg = config.get("feature_extraction", {})

    logger.info(f"[database] images: {len(images)}")
    logger.info(f"[database] max image size: {max_image}")
    logger.info(f"[database] max features: {max_features}")

    # --------------------------------------------------
    # COLMAP feature extraction
    # --------------------------------------------------

    tools.run(
        "colmap",
        [
            "feature_extractor",

            "--database_path", str(db_path),
            "--image_path", str(images_dir),

            "--ImageReader.camera_model",
            cam_cfg.get("model", "PINHOLE"),

            "--ImageReader.single_camera",
            "1" if cam_cfg.get("single", True) else "0",

            "--SiftExtraction.max_image_size",
            str(max_image),

            "--SiftExtraction.max_num_features",
            str(max_features),

            "--SiftExtraction.edge_threshold",
            str(feat_cfg.get("edge_threshold", 10)),
        ],
    )

    # --------------------------------------------------
    # Database validation
    # --------------------------------------------------

    if not db_path.exists():
        raise RuntimeError("COLMAP database was not created")

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM images")
    image_count = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM keypoints")
    kp_count = cur.fetchone()[0]

    conn.close()

    if image_count == 0:
        raise RuntimeError("COLMAP database contains zero images")

    if kp_count == 0:
        raise RuntimeError("COLMAP extracted zero keypoints")

    logger.info(f"[database] images in db: {image_count}")
    logger.info(f"[database] keypoints: {kp_count}")

    logger.info("[database] stage completed")