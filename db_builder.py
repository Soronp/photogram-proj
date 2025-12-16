#!/usr/bin/env python3
"""
database_builder.py

MARK-2: Creates and populates the COLMAP database.

Responsibilities:
- Create database/database.db
- Run feature extraction with robust parameters
- Configure camera model from runtime config
- Ensure consistent feature extraction across images

Reads:
- images_processed/

Writes:
- database/database.db
- logs/database_builder.log
"""

import subprocess
import sqlite3
from pathlib import Path

from utils.logger import get_logger
from utils.paths import ProjectPaths
from config_manager import create_runtime_config, validate_config

# --------------------------------------------------
# Helper: Run subprocess commands
# --------------------------------------------------
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
            text=True,
        )
        logger.info(result.stdout)
    except subprocess.CalledProcessError as e:
        logger.error(f"[FAIL] {label}")
        logger.error(e.stdout)
        raise RuntimeError(f"{label} failed") from e

# --------------------------------------------------
# Helper: Check image dimensions consistency
# --------------------------------------------------
def check_image_dimensions(images_dir: Path, logger):
    from PIL import Image

    logger.info("Checking image dimensions consistency...")
    image_files = sorted(f for f in images_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not image_files:
        raise RuntimeError(f"No image files found in {images_dir}")

    dimensions = set()
    for img_path in image_files[:5]:
        try:
            with Image.open(img_path) as img:
                dimensions.add(img.size)
                logger.info(f"  {img_path.name}: {img.size[0]}x{img.size[1]}")
        except Exception as e:
            logger.warning(f"Could not read {img_path.name}: {e}")

    if len(dimensions) > 1:
        logger.warning(f"Multiple image dimensions detected: {dimensions}")
        logger.warning("This may cause camera parameter issues in reconstruction")

    return len(dimensions) == 1

# --------------------------------------------------
# Core stage: Build COLMAP database
# --------------------------------------------------
def run(project_root: Path, force: bool = False):
    paths = ProjectPaths(project_root)
    paths.ensure_all()

    logger = get_logger("database_builder", project_root)

    # Load runtime config
    config = create_runtime_config(project_root)
    if not validate_config(config, logger):
        logger.warning("Config validation failed — proceeding with defaults")

    db_path = paths.database / "database.db"

    # Skip existing database if not forcing
    if db_path.exists() and not force:
        logger.info("database.db exists; skipping")
        return

    # Force rebuild
    if db_path.exists():
        logger.info("Force rebuild: removing existing database")
        db_path.unlink()

    # Read images from images_processed instead of images_filtered
    images_dir = paths.images_processed
    if not images_dir.exists() or not any(images_dir.iterdir()):
        raise FileNotFoundError(f"Processed images folder missing or empty: {images_dir}")

    # Camera parameters from config
    camera_model = config.get("camera", {}).get("model", "PINHOLE")
    single_camera = config.get("camera", {}).get("single", True)
    max_features = config.get("feature_extraction", {}).get("max_num_features", 8192)
    edge_threshold = config.get("feature_extraction", {}).get("edge_threshold", 10)

    logger.info("Starting database building stage")
    logger.info(f"Images directory : {images_dir}")
    logger.info(f"Database path    : {db_path}")
    logger.info(f"Camera model     : {camera_model}")
    logger.info(f"Single camera    : {single_camera}")
    logger.info(f"Max features     : {max_features}")
    logger.info(f"Edge threshold   : {edge_threshold}")
    logger.info(f"Force rebuild    : {force}")

    # Check image dimensions consistency
    check_image_dimensions(images_dir, logger)

    # Run COLMAP feature extraction
    run_command(
        [
            "colmap", "feature_extractor",
            "--database_path", str(db_path),
            "--image_path", str(images_dir),
            "--ImageReader.camera_model", camera_model,
            "--ImageReader.single_camera", "1" if single_camera else "0",
            "--SiftExtraction.max_image_size", "2000",
            "--SiftExtraction.max_num_features", str(max_features),
            "--SiftExtraction.edge_threshold", str(edge_threshold),
            "--SiftExtraction.first_octave", "-1",
            "--SiftExtraction.num_octaves", "4",
            "--SiftExtraction.octave_resolution", "3",
            "--SiftExtraction.peak_threshold", "0.0067",
            "--SiftExtraction.estimate_affine_shape", "0",
            "--SiftExtraction.domain_size_pooling", "0",
        ],
        logger,
        "Feature Extraction",
    )

    # --------------------------------------------------
    # Database validation
    # --------------------------------------------------
    if not db_path.exists():
        raise RuntimeError(f"Database file was not created: {db_path}")

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM images")
        image_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM keypoints")
        keypoint_count = cursor.fetchone()[0]

        conn.close()

        logger.info(f"Database validation: {image_count} images, {keypoint_count} keypoint records")
        if image_count == 0:
            raise RuntimeError("Database has no images")
        if keypoint_count == 0:
            raise RuntimeError("No keypoints extracted")
    except sqlite3.Error as e:
        logger.error(f"Failed to validate database: {e}")
        raise

    logger.info("Database build complete")

# --------------------------------------------------
# CLI wrapper
# --------------------------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(description="MARK-2 COLMAP Database Builder")
    parser.add_argument("--project", required=True, help="Project root directory")
    parser.add_argument("--force", action="store_true", help="Force rebuild database")
    args = parser.parse_args()

    run(Path(args.project), args.force)


if __name__ == "__main__":
    main()
