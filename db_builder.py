#!/usr/bin/env python3
"""
database_builder.py

Creates and populates the COLMAP database.

Responsibilities:
- Create database/database.db
- Run feature extraction with robust parameters
- Configure camera model deterministically
- Ensure consistent feature extraction across all images

Reads:
- images_processed/

Writes:
- database/database.db
- logs/database_builder.log
"""

import argparse
import subprocess
import sqlite3
from pathlib import Path

from utils.logger import get_logger
from utils.paths import ProjectPaths
from utils.config import load_config


def run_command(cmd, logger, label):
    """Run command with logging (consistent with runner pipeline)."""
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


def parse_args():
    parser = argparse.ArgumentParser(description="Build COLMAP database")
    parser.add_argument("--project", required=True, help="Path to project root")
    parser.add_argument("--force", action="store_true", help="Rebuild database")
    return parser.parse_args()


def check_image_dimensions(images_dir: Path, logger):
    """Check that all images have consistent dimensions."""
    from PIL import Image
    
    logger.info("Checking image dimensions consistency...")
    
    image_files = sorted([f for f in images_dir.iterdir() if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}])
    
    if not image_files:
        raise RuntimeError(f"No image files found in {images_dir}")
    
    dimensions = set()
    for img_path in image_files[:5]:  # Check first 5 images
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


def run(project_root: Path, force: bool = False):
    paths = ProjectPaths(project_root)
    paths.ensure_all()

    logger = get_logger("database_builder", project_root)
    config = load_config(project_root)

    db_path = paths.database / "database.db"
    
    # Skip if database exists and not forcing
    if db_path.exists() and not force:
        logger.info("database.db exists; skipping")
        
        # Quick validation that database has features
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM images")
            image_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM keypoints")
            keypoint_count = cursor.fetchone()[0]
            
            # Check image dimensions in database
            cursor.execute("SELECT width, height, COUNT(*) as count FROM images GROUP BY width, height")
            dim_results = cursor.fetchall()
            
            conn.close()
            
            logger.info(f"Database validation: {image_count} images, {keypoint_count} keypoint records")
            
            if len(dim_results) > 1:
                logger.warning("Multiple image dimensions in database:")
                for width, height, count in dim_results:
                    logger.warning(f"  {width}x{height}: {count} images")
            
        except Exception as e:
            logger.warning(f"Could not verify database contents: {e}")
        
        return

    # Remove existing database if forcing rebuild
    if db_path.exists():
        logger.warning("Removing existing database")
        db_path.unlink()

    images_dir = paths.images_processed

    camera = config.get("camera", {}).get("model", "PINHOLE")
    single = config.get("camera", {}).get("single", True)

    logger.info("Starting database building stage")
    logger.info(f"Images directory : {images_dir}")
    logger.info(f"Database path    : {db_path}")
    logger.info(f"Camera model     : {camera}")
    logger.info(f"Single camera    : {single}")
    logger.info(f"Force rebuild    : {force}")

    # --------------------------------------------------
    # Precondition: Check images directory exists
    # --------------------------------------------------
    if not images_dir.exists() or not any(images_dir.iterdir()):
        raise FileNotFoundError(
            f"Preprocessed images directory missing or empty: {images_dir}"
        )

    # --------------------------------------------------
    # Check image dimensions consistency
    # --------------------------------------------------
    check_image_dimensions(images_dir, logger)

    # --------------------------------------------------
    # Run feature extraction with ROBUST parameters
    # --------------------------------------------------
    run_command(
        [
            "colmap", "feature_extractor",
            "--database_path", str(db_path),
            "--image_path", str(images_dir),
            "--ImageReader.camera_model", camera,
            "--ImageReader.single_camera", "1" if single else "0",
            # CRITICAL: Consistent feature extraction parameters
            "--SiftExtraction.max_image_size", "2000",           # Match preprocessing
            "--SiftExtraction.max_num_features", "8192",         # Consistent limit
            "--SiftExtraction.first_octave", "-1",               # Start from original size
            "--SiftExtraction.num_octaves", "4",                 # Standard
            "--SiftExtraction.octave_resolution", "3",           # Standard
            "--SiftExtraction.peak_threshold", "0.0067",         # Default
            "--SiftExtraction.edge_threshold", "10",             # Default
            "--SiftExtraction.estimate_affine_shape", "0",       # Disable (more consistent)
            "--SiftExtraction.domain_size_pooling", "0",         # Disable (more consistent)
        ],
        logger,
        "Feature Extraction",
    )

    # --------------------------------------------------
    # Validate database was created successfully
    # --------------------------------------------------
    if not db_path.exists():
        raise RuntimeError(f"Database file was not created: {db_path}")

    # Quick validation
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check images were added
        cursor.execute("SELECT COUNT(*) FROM images")
        image_count = cursor.fetchone()[0]
        
        # Check keypoints were extracted
        cursor.execute("SELECT COUNT(*) FROM keypoints")
        keypoint_count = cursor.fetchone()[0]
        
        # Check descriptor consistency
        cursor.execute("SELECT rows, cols FROM descriptors WHERE image_id = 1")
        desc_result = cursor.fetchone()
        
        conn.close()
        
        logger.info("Database validation:")
        logger.info(f"  Images in database: {image_count}")
        logger.info(f"  Keypoint records: {keypoint_count}")
        
        if desc_result:
            rows, cols = desc_result
            logger.info(f"  Descriptor dimensions: {rows}x{cols} (image 1)")
        
        if image_count == 0:
            logger.error("No images found in database!")
            raise RuntimeError("Database has no images")
            
        if keypoint_count == 0:
            logger.error("No keypoints extracted!")
            raise RuntimeError("Database has no keypoints")
            
    except sqlite3.Error as e:
        logger.error(f"Failed to validate database: {e}")
        raise

    logger.info("Database build complete")


def main():
    args = parse_args()
    run(Path(args.project), args.force)


if __name__ == "__main__":
    main()