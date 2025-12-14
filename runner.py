#!/usr/bin/env python3
"""
pipeline_runner.py

MARK-2 Pipeline Runner (Clean & Re-run)
-------------------------------------
- Always cleans stage outputs before running
- Enforces strict execution order
- No skip logic (deterministic, reproducible)
- Safe to re-run at any time

This runner intentionally trades speed for correctness.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from datetime import datetime

from utils.logger import get_logger
from utils.paths import ProjectPaths


# Ordered pipeline (current core)
PIPELINE_STEPS = [
    "input.py",
    "filter.py",
    "db_builder.py",
    "matcher.py",
    "sparse_reconstruction.py",
    "sparse_eval.py",
    "dense_reconstruction.py",
    "dense_eval.py",
    "gen_mesh.py",
]


# Outputs to clean BEFORE running each step
CLEAN_TARGETS = {
    "input.py": ["images"],
    "filter.py": ["images_filtered"],
    "db_builder.py": ["database"],
    "matcher.py": [],  # matcher writes into database
    "sparse_reconstruction.py": ["sparse"],
    "sparse_eval.py": ["evaluation/sparse_metrics.json"],
    "dense_reconstruction.py": ["dense"],
    "dense_eval.py": ["evaluation/dense_metrics.json"],
    "gen_mesh.py": ["mesh"],
}


# Scripts that use different argument formats
# Add scripts here that don't follow the standard "--project" format
POSITIONAL_ARG_SCRIPTS = [
    "sparse_reconstruction.py",  # Uses: project_root (positional)
    # Add other scripts here if they also use positional args
]


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def rm_path(path: Path, logger):
    if not path.exists():
        return

    if path.is_file():
        logger.warning(f"Removing file: {path}")
        path.unlink()
    else:
        logger.warning(f"Removing directory: {path}")
        shutil.rmtree(path)


def clean_step(step: str, project_root: Path, logger):
    targets = CLEAN_TARGETS.get(step, [])
    for rel in targets:
        rm_path(project_root / rel, logger)


def run_step(step: str, project_root: Path, logger):
    """Run a pipeline step with appropriate arguments."""
    
    # Build command based on script's argument format
    if step in POSITIONAL_ARG_SCRIPTS:
        # Scripts that take project_root as positional argument
        cmd = [sys.executable, step, str(project_root), "--force"]
    else:
        # Standard scripts that use --project flag
        cmd = [sys.executable, step, "--project", str(project_root), "--force"]

    logger.info(f"[RUN] {' '.join(cmd)}")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        logger.error(f"[FAIL] {step} exited with code {result.returncode}")
        sys.exit(result.returncode)

    logger.info(f"[DONE] {step}")


# --------------------------------------------------
# Main
# --------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="MARK-2 clean pipeline runner")
    parser.add_argument("--project", required=True, help="Path to project root")
    parser.add_argument("--stop-after", help="Stop after a given step (debug)")
    return parser.parse_args()


def main():
    args = parse_args()

    project_root = Path(args.project).resolve()
    paths = ProjectPaths(project_root)
    paths.ensure_all()

    logger = get_logger("pipeline", project_root)

    logger.info("Starting CLEAN pipeline run")
    logger.info(f"Project root: {project_root}")
    logger.info("Mode: destructive clean + rebuild")

    start = datetime.utcnow()

    for step in PIPELINE_STEPS:
        logger.info(f"[CLEAN] {step}")
        clean_step(step, project_root, logger)

        run_step(step, project_root, logger)

        if args.stop_after == step:
            logger.warning(f"Stopping early after {step}")
            break

    elapsed = datetime.utcnow() - start
    logger.info(f"Pipeline finished in {elapsed}")


if __name__ == "__main__":
    main()