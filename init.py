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


def parse_args():
    parser = argparse.ArgumentParser(description="Initialize photogrammetry project")
    parser.add_argument("--project", required=True, help="Path to project root")
    parser.add_argument("--name", help="Project name (defaults to folder name)")
    parser.add_argument("--input-type", choices=["images", "video"], default="images")
    parser.add_argument("--force", action="store_true", help="Overwrite existing project.json")
    return parser.parse_args()


def main():
    args = parse_args()

    project_root = Path(args.project).resolve()
    project_root.mkdir(parents=True, exist_ok=True)

    # Initialize paths
    paths = ProjectPaths(project_root)
    paths.ensure_all()

    # Initialize logger early
    logger = get_logger("init", project_root)
    logger.info("Initializing project")
    logger.info(f"Project root: {project_root}")

    # Load config (must exist)
    try:
        config = load_config(project_root)
        logger.info("Loaded config.yaml")
    except FileNotFoundError:
        logger.error("config.yaml not found in project root")
        raise

    # Write project.json
    project_json_path = project_root / "project.json"

    if project_json_path.exists() and not args.force:
        logger.warning("project.json already exists; use --force to overwrite")
    else:
        project_manifest = {
            "project_name": args.name or project_root.name,
            "created": datetime.now().isoformat() + "Z",
            "input_type": args.input_type,
            "project_root": str(project_root),
            "structure": REQUIRED_DIRS,
        }

        with open(project_json_path, "w", encoding="utf-8") as f:
            json.dump(project_manifest, f, indent=2)

        logger.info("Wrote project.json")

    # Snapshot config for reproducibility
    config_snapshot = project_root / "logs" / "config_snapshot.yaml"
    try:
        shutil.copy(project_root / "config.yaml", config_snapshot)
        logger.info("Saved config snapshot")
    except Exception as e:
        logger.warning(f"Could not snapshot config.yaml: {e}")

    logger.info("Project initialization complete")


if __name__ == "__main__":
    main()
