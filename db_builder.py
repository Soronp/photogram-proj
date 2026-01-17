#!/usr/bin/env python3
"""
database_builder_cpu_safe.py

MARK-2: COLMAP Database Builder (CPU-only, ToolRunner-managed)
-------------------------------------------------------------
- Forces CPU extraction to prevent GPU OOM
- Auto-adjusts SIFT parameters
- Resume-safe, force-aware
- Runner-injected logger ONLY
"""

import os
import sqlite3
from pathlib import Path
from PIL import Image

from utils.paths import ProjectPaths
from config_manager import create_runtime_config, validate_config
from tool_runner import ToolRunner

# --------------------------------------------------
# CONSTANTS
# --------------------------------------------------
VALID_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

# FORCE CPU — module-level, explicit
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# --------------------------------------------------
# INTERNAL HELPERS (PURE, NO SIDE EFFECTS)
# --------------------------------------------------
def collect_images(images_dir: Path):
    return sorted(
        f for f in images_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in VALID_EXTS
    )

def determine_max_image_size(image_files, logger) -> int:
    dims = [Image.open(p).size for p in image_files]
    max_dim = min(max(max(w, h) for w, h in dims), 3072)
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
# PIPELINE STAGE ENTRYPOINT
# --------------------------------------------------
def run(run_root: Path, project_root: Path, force: bool, logger):
    """
    MARK-2 pipeline stage: database builder
    """

    paths = ProjectPaths(project_root)
    paths.ensure_all()

    logger.info("[database] Stage started")

    # --------------------------------------------------
    # CONFIG (IMMUTABLE PER RUN)
    # --------------------------------------------------
    config = create_runtime_config(run_root, project_root, logger)
    if not validate_config(config, logger):
        raise RuntimeError("Invalid configuration")

    tool = ToolRunner(config, logger)

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

    cam_cfg = config.get("camera", {})
    feat_cfg = config.get("feature_extraction", {})

    camera_model = cam_cfg.get("model", "PINHOLE")
    single_camera = cam_cfg.get("single", True)
    edge_threshold = feat_cfg.get("edge_threshold", 10)

    logger.info(f"[database] Images        : {num_images}")
    logger.info(f"[database] Max features  : {max_features}")
    logger.info(f"[database] Camera model  : {camera_model}")
    logger.info(f"[database] Single camera : {single_camera}")

    # --------------------------------------------------
    # FEATURE EXTRACTION (COLMAP, CPU)
    # --------------------------------------------------
    tool.run(
        tool="colmap",
        args=[
            "feature_extractor",
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
        cwd=project_root,
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

    logger.info(
        f"[database] Database ready — "
        f"{image_count} images, {keypoint_count} keypoints"
    )
    logger.info("[database] Stage completed successfully")
