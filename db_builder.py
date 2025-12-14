#!/usr/bin/env python3
"""
database_builder.py

Creates and populates the COLMAP database.

Responsibilities:
- Create database/database.db
- Run feature extraction
- Configure camera model deterministically

Reads:
- images_filtered/

Writes:
- database/database.db
- logs/database_builder.log
"""

import argparse
import subprocess
from pathlib import Path

from utils.logger import get_logger
from utils.paths import ProjectPaths
from utils.config import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Build COLMAP database")
    parser.add_argument("--project", required=True, help="Path to project root")
    parser.add_argument("--force", action="store_true", help="Rebuild database")
    return parser.parse_args()


def run(cmd, logger):
    logger.info("Running: " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    project_root = Path(args.project).resolve()

    paths = ProjectPaths(project_root)
    paths.ensure_all()

    logger = get_logger("database_builder", project_root)
    config = load_config(project_root)

    db_path = paths.database / "database.db"

    if db_path.exists() and not args.force:
        logger.info("database.db already exists; skipping")
        return

    if db_path.exists() and args.force:
        db_path.unlink()

    images_dir = paths.images_filtered

    camera_model = config.get("camera", {}).get("model", "PINHOLE")
    single_camera = config.get("camera", {}).get("single", True)

    logger.info(f"Camera model: {camera_model}")
    logger.info(f"Single camera: {single_camera}")

    cmd = [
        "colmap", "feature_extractor",
        "--database_path", str(db_path),
        "--image_path", str(images_dir),
        "--ImageReader.camera_model", camera_model,
        "--ImageReader.single_camera", "1" if single_camera else "0",
    ]

    run(cmd, logger)

    logger.info("Database build complete")


if __name__ == "__main__":
    main()
