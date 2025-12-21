#!/usr/bin/env python3
"""
init.py

MARK-2 Output / Project Initialization
--------------------------------------
- Initializes tool-agnostic output directories under project root
- Writes config.yaml + output.json
- Accepts logger for pipeline parity
"""

from pathlib import Path
from datetime import datetime
import json
import yaml

from utils.paths import ProjectPaths


def run(run_root: Path, project_root: Path, force: bool, logger):
    """
    Initialize project output directories and config.
    Compatible with MARK-2 pipeline runner (accepts logger).
    """
    project_root = project_root.resolve()
    paths = ProjectPaths(project_root)
    paths.ensure_all()

    # Use canonical tool-agnostic evaluation folder as "output"
    output_root = paths.evaluation
    output_root.mkdir(parents=True, exist_ok=True)

    msg = f"[init] Initializing output directory: {output_root}"
    if logger:
        logger.info(msg)
    else:
        print(msg)

    # --- Write config.yaml ---
    config_path = output_root / "config.yaml"
    if force or not config_path.exists():
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                {
                    "input_type": "images",
                    "matcher_type": "sequential",
                    "dense_quality": "medium",
                },
                f,
            )
        msg = f"[init] config.yaml written: {config_path}"
        if logger:
            logger.info(msg)
        else:
            print(msg)
    else:
        msg = f"[init] config.yaml exists — skipping"
        if logger:
            logger.info(msg)
        else:
            print(msg)

    # --- Write output.json ---
    manifest_path = output_root / "output.json"
    manifest = {
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "project_root": str(project_root),
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    msg = f"[init] output.json written: {manifest_path}"
    if logger:
        logger.info(msg)
    else:
        print(msg)

    msg = "[init] Output initialization complete"
    if logger:
        logger.info(msg)
    else:
        print(msg)


# --- CLI entrypoint ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MARK-2 Output / Project Initializer")
    parser.add_argument("--project", required=True, help="Project root directory")
    parser.add_argument("--force", action="store_true", help="Force re-run")
    args = parser.parse_args()

    run(Path(args.project), args.force)
