#!/usr/bin/env python3
"""
init.py

Project initialization stage (MARK-2)
"""

import argparse
import json
import shutil
from pathlib import Path
from datetime import datetime

from utils.logger import get_logger
from utils.config import load_config
from utils.paths import ProjectPaths


REQUIRED_DIRS = [
    "raw",
    "images",
    "images_filtered",
    "database",
    "sparse",
    "dense",
    "mesh",
    "textures",
    "evaluation",
    "logs",
]


# --------------------------------
# CORE CALLABLE (USED BY RUNNER)
# --------------------------------

def run_init(project_root: Path, force: bool = False):
    project_root = project_root.resolve()
    project_root.mkdir(parents=True, exist_ok=True)

    paths = ProjectPaths(project_root)
    paths.ensure_all()

    logger = get_logger("init", project_root)
    logger.info("Initializing project")
    logger.info(f"Project root: {project_root}")

    # Load config (required)
    try:
        config = load_config(project_root)
        logger.info("Loaded config.yaml")
    except FileNotFoundError:
        logger.error("config.yaml not found in project root")
        raise

    project_json_path = project_root / "project.json"

    if project_json_path.exists() and not force:
        logger.warning("project.json already exists; use --force to overwrite")
    else:
        project_manifest = {
            "project_name": project_root.name,
            "created": datetime.utcnow().isoformat() + "Z",
            "input_type": config.get("input_type", "images"),
            "project_root": str(project_root),
            "structure": REQUIRED_DIRS,
        }

        with open(project_json_path, "w", encoding="utf-8") as f:
            json.dump(project_manifest, f, indent=2)

        logger.info("Wrote project.json")

    # Snapshot config
    try:
        shutil.copy(project_root / "config.yaml",
                    project_root / "logs" / "config_snapshot.yaml")
        logger.info("Saved config snapshot")
    except Exception as e:
        logger.warning(f"Could not snapshot config.yaml: {e}")

    logger.info("Project initialization complete")


# --------------------------------
# CLI WRAPPER (OPTIONAL)
# --------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Initialize photogrammetry project")
    parser.add_argument("--project", required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    run_init(Path(args.project), args.force)


if __name__ == "__main__":
    main()
