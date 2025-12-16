#!/usr/bin/env python3
"""
evaluation_aggregator.py

MARK-2 Evaluation Aggregation Stage
-----------------------------------
- Aggregates evaluation metrics from all stages
- Handles missing metrics gracefully
- Produces unified JSON and CSV summaries
- Deterministic and restart-safe
- Pipeline-compatible: run_evaluation_aggregation(project_root, force)
"""

import argparse
import json
import csv
from pathlib import Path

from utils.logger import get_logger
from utils.paths import ProjectPaths
from utils.config import load_config


# ------------------------------------------------------------------
# Aggregation logic
# ------------------------------------------------------------------

def run_evaluation_aggregation(project_root: Path, force: bool = False):
    paths = ProjectPaths(project_root)
    _ = load_config(project_root)
    logger = get_logger("evaluation_aggregator", project_root)

    eval_dir = paths.evaluation
    summary_json = eval_dir / "summary.json"
    summary_csv = eval_dir / "summary.csv"

    logger.info("=== Evaluation Aggregation Stage ===")
    logger.info(f"Evaluation dir : {eval_dir}")

    if not eval_dir.exists():
        raise FileNotFoundError("Evaluation directory does not exist")

    # Known metric files (extendable)
    metric_files = {
        "sparse": eval_dir / "sparse_metrics.json",
        "dense": eval_dir / "dense_metrics.json",
        "mesh": eval_dir / "mesh_metrics.json",
    }

    aggregated = {
        "project_root": str(project_root),
        "stages": {},
        "deterministic": True,
    }

    flat_rows = []

    # --------------------------------------------------
    # Load available metrics
    # --------------------------------------------------

    for stage, path in metric_files.items():
        if not path.exists():
            logger.warning(f"Metrics missing for stage '{stage}' — skipping")
            continue

        logger.info(f"Loading metrics: {path.name}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        aggregated["stages"][stage] = data

        # Flatten for CSV
        for key, value in data.items():
            if isinstance(value, (int, float, str, bool)):
                flat_rows.append({
                    "stage": stage,
                    "metric": key,
                    "value": value,
                })

    if not aggregated["stages"]:
        logger.warning("No evaluation metrics found — summary will be empty")

    # --------------------------------------------------
    # Write summary.json
    # --------------------------------------------------

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(aggregated, f, indent=2)

    logger.info(f"Summary JSON written to: {summary_json}")

    # --------------------------------------------------
    # Write summary.csv
    # --------------------------------------------------

    if flat_rows:
        with open(summary_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["stage", "metric", "value"]
            )
            writer.writeheader()
            writer.writerows(flat_rows)

        logger.info(f"Summary CSV written to: {summary_csv}")
    else:
        logger.warning("No scalar metrics available for CSV export")

    logger.info("Evaluation aggregation completed successfully")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MARK-2 Evaluation Aggregator")
    parser.add_argument("project_root", type=Path)
    parser.add_argument("--force", action="store_true", help="Force re-aggregation")
    args = parser.parse_args()

    run_evaluation_aggregation(args.project_root, args.force)


if __name__ == "__main__":
    main()
