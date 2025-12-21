#!/usr/bin/env python3
"""
database_builder_cpu_safe.py

MARK-2: COLMAP Database Builder (CPU-only, runner-managed)
---------------------------------------------------------
- Forces CPU extraction to prevent GPU OOM
- Auto-adjusts SIFT parameters
- Resume-safe, force-aware
- Runner-injected logger ONLY
"""

import os
import subprocess
import sqlite3
from pathlib import Path
from PIL import Image

from utils.paths import ProjectPaths
from config_manager import create_runtime_config, validate_config

# --------------------------------------------------
# CONSTANTS
# --------------------------------------------------
VALID_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

# FORCE CPU — must be module-level
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# --------------------------------------------------
# INTERNAL HELPERS (NO LOGGER CREATION)
# --------------------------------------------------
def run_command(cmd, logger, label: str):
    cmd = [str(c) for c in cmd]
    logger.info(f"[database] RUN: {label}")
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
        logger.error(f"[database] FAILED: {label}")
        logger.error(e.stdout)
        raise RuntimeError(f"{label} failed") from e

def collect_images(images_dir: Path):
    return sorted(
        f for f in images_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in VALID_EXTS
    )

def determine_max_image_size(image_files, logger) -> int:
    dims = [Image.open(p).size for p in image_files]
    max_w = max(w for w, _ in dims)
    max_h = max(h for _, h in dims)
    max_dim = min(max(max_w, max_h), 3072)

    logger.info(f"[database] Computed SIFT max_image_size = {max_dim}")
    return max_dim

def auto_max_features(num_images: int) -> int:
    if num_images < 80:
        return 20000
    if num_images < 150:
        return 18000
    if num_images < 300:
        return 15000
    if num_images < 800:
        return 12000
    return 9000

# --------------------------------------------------
# PIPELINE STAGE ENTRYPOINT (STRICT CONTRACT)
# --------------------------------------------------
def run(run_root: Path, project_root: Path, force: bool, logger):
    """
    MARK-2 pipeline stage: database builder

    Contract:
    - project_root is authoritative
    - logger is injected by runner
    - raise on failure
    """

    paths = ProjectPaths(project_root)
    paths.ensure_all()

    logger.info("[database] Stage started")

    # --------------------------------------------------
    # CONFIG
    # --------------------------------------------------
    config = create_runtime_config(run_root, project_root, logger)
    validate_config(config, logger)

    # --------------------------------------------------
    # DATABASE HANDLING
    # --------------------------------------------------
    db_path = paths.database / "database.db"

    if db_path.exists():
        if not force:
            logger.info("[database] Existing database detected — skipping")
            return
        logger.info("[database] Force enabled — removing existing database")
        db_path.unlink()

    # --------------------------------------------------
    # IMAGE DISCOVERY
    # --------------------------------------------------
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

    logger.info(f"[database] Images        : {num_images}")
    logger.info(f"[database] Max features  : {max_features}")
    logger.info(f"[database] Camera model  : {camera_model}")
    logger.info(f"[database] Single camera : {single_camera}")

    # --------------------------------------------------
    # FEATURE EXTRACTION (CPU)
    # --------------------------------------------------
    run_command(
        [
            "colmap", "feature_extractor",
            "--database_path", db_path,
            "--image_path", images_dir,
            "--ImageReader.camera_model", camera_model,
            "--ImageReader.single_camera", "1" if single_camera else "0",
            "--SiftExtraction.max_image_size", max_image_size,
            "--SiftExtraction.max_num_features", max_features,
            "--SiftExtraction.edge_threshold", edge_threshold,
            "--SiftExtraction.first_octave", "-1",
            "--SiftExtraction.num_octaves", "5",
            "--SiftExtraction.octave_resolution", "3",
            "--SiftExtraction.peak_threshold", "0.006",
        ],
        logger,
        "Feature Extraction (CPU)"
    )

    # --------------------------------------------------
    # DATABASE VALIDATION
    # --------------------------------------------------
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM images")
    image_count = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM keypoints")
    keypoint_count = cur.fetchone()[0]
    conn.close()

    if image_count == 0 or keypoint_count == 0:
        raise RuntimeError("Feature extraction produced empty database")

    logger.info(f"[database] Database ready — {image_count} images, {keypoint_count} keypoints")
    logger.info("[database] Stage completed successfully")
