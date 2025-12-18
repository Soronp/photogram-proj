#!/usr/bin/env python3
"""
database_builder_cpu_safe.py

MARK-2: COLMAP Database Builder (GPU-safe, CPU only)
---------------------------------------------------
- Forces CPU extraction to prevent GPU OOM
- Removes obsolete flags
- Auto-adjusts max features
"""

import subprocess
import sqlite3
from pathlib import Path
from PIL import Image
import os

from utils.logger import get_logger
from utils.paths import ProjectPaths
from config_manager import create_runtime_config, validate_config

VALID_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

# ----- FORCE CPU -----
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Prevent COLMAP from using GPU

def run_command(cmd, logger, label):
    cmd = [str(c) for c in cmd]
    logger.info(f"[RUN] {label}")
    logger.info(" ".join(cmd))
    try:
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        logger.info(result.stdout)
    except subprocess.CalledProcessError as e:
        logger.error(f"[FAIL] {label}")
        logger.error(e.stdout)
        raise RuntimeError(f"{label} failed") from e

def collect_images(images_dir: Path):
    return sorted(
        f for f in images_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in VALID_EXTS
    )

def determine_max_image_size(image_files, logger) -> int:
    dims = [Image.open(img_path).size for img_path in image_files]
    max_w = max(w for w, h in dims)
    max_h = max(h for w, h in dims)
    max_dim = min(max(max_w, max_h), 3072)  # Limit for memory safety
    logger.info(f"Computed max image dimension for SIFT: {max_dim}")
    return max_dim

def auto_max_features(num_images: int) -> int:
    if num_images < 80:
        return 20000
    elif num_images < 150:
        return 18000
    elif num_images < 300:
        return 15000
    elif num_images < 800:
        return 12000
    else:
        return 9000

def run(project_root: Path, force: bool = False):
    paths = ProjectPaths(project_root)
    paths.ensure_all()
    logger = get_logger("database_builder", project_root)

    config = create_runtime_config(project_root)
    validate_config(config, logger)

    db_path = paths.database / "database.db"
    if db_path.exists():
        if not force:
            logger.info("Database exists; skipping")
            return
        logger.info("Force rebuild — removing database")
        db_path.unlink()

    images_dir = paths.images_processed
    image_files = collect_images(images_dir)
    if not image_files:
        raise RuntimeError("No processed images found")

    num_images = len(image_files)
    max_image_size = determine_max_image_size(image_files, logger)
    max_features = auto_max_features(num_images)
    edge_threshold = config.get("feature_extraction", {}).get("edge_threshold", 10)
    camera_model = config.get("camera", {}).get("model", "PINHOLE")
    single_camera = config.get("camera", {}).get("single", True)

    logger.info(f"Images: {num_images}")
    logger.info(f"Max features: {max_features}")

    # ---------------- FEATURE EXTRACTION ----------------
    run_command(
        [
            "colmap", "feature_extractor",
            "--database_path", str(db_path),
            "--image_path", str(images_dir),

            "--ImageReader.camera_model", camera_model,
            "--ImageReader.single_camera", "1" if single_camera else "0",

            "--SiftExtraction.max_image_size", str(max_image_size),
            "--SiftExtraction.max_num_features", str(max_features),
            "--SiftExtraction.edge_threshold", str(edge_threshold),

            "--SiftExtraction.first_octave", "-1",
            "--SiftExtraction.num_octaves", "5",
            "--SiftExtraction.octave_resolution", "3",
            "--SiftExtraction.peak_threshold", "0.006",
        ],
        logger,
        "Feature Extraction (CPU)"
    )

    # ---------------- DATABASE VALIDATION ----------------
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM images")
    image_count = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM keypoints")
    keypoint_count = cur.fetchone()[0]
    conn.close()

    if image_count == 0 or keypoint_count == 0:
        raise RuntimeError("Feature extraction produced empty database")

    logger.info(f"Database ready: {image_count} images, {keypoint_count} keypoints")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run(Path(args.project), args.force)
