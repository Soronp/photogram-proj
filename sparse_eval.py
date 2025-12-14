#!/usr/bin/env python3
"""
sparse_evaluation.py

MARK-2 Sparse Evaluation Stage
-----------------------------
- Evaluates sparse reconstruction quality using COLMAP model_analyzer
- Extracts key metrics for pipeline health and comparison
- Writes JSON + CSV for downstream aggregation
- Fully logged, deterministic, restart-safe
"""

import argparse
import json
import csv
import subprocess
import re
from pathlib import Path

from utils.logger import get_logger
from utils.paths import ProjectPaths
from utils.config import load_config, COLMAP_EXE


# ------------------------------------------------------------------
# Command runner
# ------------------------------------------------------------------

def run_command(cmd, logger, label):
    cmd = [str(c) for c in cmd]
    logger.info(f"[RUN] {label}")
    logger.info(" ".join(cmd))

    result = subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    logger.info(result.stdout)
    return result.stdout


# ------------------------------------------------------------------
# Metric parsing
# ------------------------------------------------------------------

def parse_model_analyzer(output: str) -> dict:
    """
    Parse key metrics from COLMAP model_analyzer output.
    Uses regex for version-robust extraction.
    """

    metrics = {}

    patterns = {
        "num_images": r"Registered images:\s+(\d+)",
        "num_points": r"Points:\s+(\d+)",
        "num_observations": r"Observations:\s+(\d+)",
        "mean_track_length": r"Mean track length:\s+([\d.]+)",
        "mean_observations_per_image": r"Mean observations per image:\s+([\d.]+)",
        "mean_reprojection_error": r"Mean reprojection error:\s+([\d.]+)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, output)
        if match:
            value = match.group(1)
            metrics[key] = float(value) if "." in value else int(value)
        else:
            metrics[key] = None

    return metrics


# ------------------------------------------------------------------
# Sparse evaluation
# ------------------------------------------------------------------

def run_sparse_evaluation(project_root: Path, force: bool):
    paths = ProjectPaths(project_root)
    config = load_config(project_root)
    logger = get_logger("sparse_evaluation", project_root)

    sparse_root = paths.sparse
    eval_dir = paths.evaluation
    eval_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting sparse reconstruction evaluation")
    logger.info(f"Sparse root: {sparse_root}")
    logger.info(f"Evaluation output: {eval_dir}")

    if not sparse_root.exists():
        raise FileNotFoundError("sparse/ directory does not exist")

    # Find sparse model folder (usually sparse/0)
    model_dir = None
    for d in sparse_root.iterdir():
        if d.is_dir() and (d / "cameras.bin").exists():
            model_dir = d
            break

    if model_dir is None:
        raise RuntimeError("No valid sparse model found for evaluation")

    metrics_json = eval_dir / "sparse_metrics.json"
    metrics_csv = eval_dir / "sparse_metrics.csv"

    if metrics_json.exists() and not force:
        logger.info("Sparse metrics already exist — skipping evaluation")
        return

    # --------------------------------------------------
    # Run COLMAP model_analyzer
    # --------------------------------------------------

    output = run_command(
        [
            COLMAP_EXE,
            "model_analyzer",
            "--path", model_dir
        ],
        logger,
        "Sparse Model Analysis"
    )

    metrics = parse_model_analyzer(output)
    metrics["model_path"] = str(model_dir)
    metrics["stage"] = "sparse"

    # --------------------------------------------------
    # Write JSON
    # --------------------------------------------------

    with open(metrics_json, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Wrote sparse metrics JSON: {metrics_json}")

    # --------------------------------------------------
    # Write CSV (single-row, aggregator-friendly)
    # --------------------------------------------------

    with open(metrics_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metrics.keys())
        writer.writeheader()
        writer.writerow(metrics)

    logger.info(f"Wrote sparse metrics CSV: {metrics_csv}")
    logger.info("Sparse evaluation completed successfully")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MARK-2 Sparse Evaluation")
    parser.add_argument("project_root", type=Path)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing sparse evaluation outputs"
    )

    args = parser.parse_args()
    run_sparse_evaluation(args.project_root, args.force)


if __name__ == "__main__":
    main()
