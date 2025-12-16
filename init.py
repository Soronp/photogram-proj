#!/usr/bin/env python3
"""
init.py

Project initialization stage (MARK-2)
-------------------------------------
- Creates project directories
- Creates project.json
- Creates a minimal base config.yaml (if missing)
- Fully compatible with interactive runner and config_manager
"""

import argparse
import json
import shutil
from pathlib import Path
from datetime import datetime
import yaml

from utils.logger import get_logger
from utils.paths import ProjectPaths

# -------------------------------
# Required directories
# -------------------------------
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

# -------------------------------
# Core callable
# -------------------------------
def run_init(project_root: Path, force: bool = False):
    """
    Initialize the project:
    - Create directories
    - Create minimal config.yaml
    - Create project.json
    """
    project_root = project_root.resolve()
    project_root.mkdir(parents=True, exist_ok=True)

    paths = ProjectPaths(project_root)
    paths.ensure_all()

    logger = get_logger("init", project_root)
    logger.info("Initializing project")
    logger.info(f"Project root: {project_root}")

    # -------------------------------
    # Minimal config.yaml
    # -------------------------------
    config_path = project_root / "config.yaml"
    if not config_path.exists() or force:
        minimal_config = {
            "input_type": "images",
            "sparse_max_features": 6000,
            "dense_quality": "medium",
            "matcher_type": "sequential",
            "image_count": 0
        }
        try:
            with open(config_path, "w") as f:
                yaml.safe_dump(minimal_config, f)
            logger.info(f"Created minimal config.yaml at {config_path}")
        except Exception as e:
            logger.error(f"Failed to create config.yaml: {e}")
            raise
    else:
        logger.info("config.yaml already exists; not overwriting")

    # -------------------------------
    # project.json
    # -------------------------------
    project_json_path = project_root / "project.json"
    if project_json_path.exists() and not force:
        logger.warning("project.json already exists; use --force to overwrite")
    else:
        project_manifest = {
            "project_name": project_root.name,
            "created": datetime.utcnow().isoformat() + "Z",
            "input_type": "images",
            "project_root": str(project_root),
            "structure": REQUIRED_DIRS,
        }
        try:
            with open(project_json_path, "w", encoding="utf-8") as f:
                json.dump(project_manifest, f, indent=2)
            logger.info("Wrote project.json")
        except Exception as e:
            logger.error(f"Failed to write project.json: {e}")
            raise

    # -------------------------------
    # Snapshot config.yaml
    # -------------------------------
    try:
        shutil.copy(config_path, project_root / "logs" / "config_snapshot.yaml")
        logger.info("Saved config snapshot")
    except Exception as e:
        logger.warning(f"Could not snapshot config.yaml: {e}")

    logger.info("Project initialization complete")

# -------------------------------
# CLI wrapper
# -------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Initialize MARK-2 photogrammetry project")
    parser.add_argument("--project", required=True, help="Project root folder")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    return parser.parse_args()

def main():
    args = parse_args()
    run_init(Path(args.project), args.force)

if __name__ == "__main__":
    main()
