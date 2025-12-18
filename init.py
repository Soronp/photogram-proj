#!/usr/bin/env python3
"""
MARK-2 project initialization
"""

import argparse
import json
import shutil
from pathlib import Path
from datetime import datetime
import yaml

from utils.logger import get_logger
from utils.paths import ProjectPaths


def run_init(project_root: Path, force: bool):
    project_root = project_root.resolve()
    project_root.mkdir(parents=True, exist_ok=True)

    paths = ProjectPaths(project_root)
    paths.ensure_all()
    paths.validate()

    logger = get_logger("init", project_root)
    logger.info("Initializing MARK-2 project")

    # -----------------------------
    # config.yaml
    # -----------------------------
    config_path = project_root / "config.yaml"
    if force or not config_path.exists():
        with open(config_path, "w") as f:
            yaml.safe_dump({
                "input_type": "images",
                "matcher_type": "sequential",
                "dense_quality": "medium",
            }, f)

    # -----------------------------
    # project.json (truthful)
    # -----------------------------
    manifest = {
        "project_name": project_root.name,
        "created": datetime.utcnow().isoformat() + "Z",
        "root": str(project_root),
        "paths": sorted(
            str(p.relative_to(project_root))
            for p in vars(paths).values()
            if isinstance(p, Path)
        ),
    }

    with open(project_root / "project.json", "w") as f:
        json.dump(manifest, f, indent=2)

    shutil.copy(config_path, paths.logs / "config_snapshot.yaml")
    logger.info("Project initialized")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", required=True)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    run_init(Path(args.project), args.force)


if __name__ == "__main__":
    main()
