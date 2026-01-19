#!/usr/bin/env python3
"""
db_builder.py

MARK-2 Database Builder (Authoritative)
--------------------------------------
- ToolRunner enforced
- GPU opt-in via config
- Resume-safe
- Deterministic
"""

import sqlite3
from pathlib import Path
from PIL import Image

from utils.paths import ProjectPaths
from config_manager import load_config
from tool_runner import ToolRunner

VALID_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def _collect_images(images_dir: Path):
    return sorted(
        f for f in images_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in VALID_EXTS
    )


def _compute_max_image_size(images):
    dims = [Image.open(p).size for p in images]
    return min(max(max(w, h) for w, h in dims), 3072)


def _auto_max_features(n):
    if n < 80:
        return 20000
    if n < 150:
        return 18000
    if n < 300:
        return 15000
    if n < 800:
        return 12000
    return 9000


def run(run_root: Path, project_root: Path, force: bool, logger):
    paths = ProjectPaths(project_root)
    paths.ensure_all()

    logger.info("[database] START")

    config = load_config(run_root, logger)
    tool = ToolRunner(config, logger)

    db_path = paths.database / "database.db"

    if db_path.exists():
        if not force:
            logger.info("[database] Existing DB detected — skipping")
            return
        logger.info("[database] Force enabled — removing DB")
        db_path.unlink()

    images = _collect_images(paths.images_processed)
    if not images:
        raise RuntimeError("No processed images found")

    max_image = _compute_max_image_size(images)
    max_features = _auto_max_features(len(images))

    cam = config.get("camera", {})
    feat = config["feature_extraction"]

    tool.run(
        tool="colmap",
        args=[
            "feature_extractor",
            "--database_path", str(db_path),
            "--image_path", str(paths.images_processed),
            "--ImageReader.camera_model", cam.get("model", "PINHOLE"),
            "--ImageReader.single_camera", "1" if cam.get("single", True) else "0",
            "--SiftExtraction.max_image_size", str(max_image),
            "--SiftExtraction.max_num_features", str(max_features),
            "--SiftExtraction.edge_threshold", str(feat.get("edge_threshold", 10)),
        ],
    )

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM images")
    cur.execute("SELECT COUNT(*) FROM keypoints")
    if cur.fetchone()[0] == 0:
        raise RuntimeError("Empty database produced")
    conn.close()

    logger.info("[database] COMPLETED")
