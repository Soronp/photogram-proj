#!/usr/bin/env python3
"""
evaluation_aggregator.py

MARK-2 Evaluation Aggregation Stage
-----------------------------------
Aggregates sparse / dense / mesh metrics.
Runner-managed, deterministic.
"""

import json
import csv
from pathlib import Path

from utils.paths import ProjectPaths
from utils.config import load_config


# --------------------------------------------------
# PIPELINE STAGE
# --------------------------------------------------

def run(project_root: Path, force: bool, logger):
    paths = ProjectPaths(project_root)
    _ = load_config(project_root)

    eval_dir = paths.evaluation
    summary_json = eval_dir / "summary.json"
    summary_csv = eval_dir / "summary.csv"

    logger.info("[eval_agg] Stage started")

    if not eval_dir.exists():
        raise FileNotFoundError("Evaluation directory does not exist")

    if summary_json.exists() and not force:
        logger.info("[eval_agg] Summary already exists — skipping")
        return

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

    for stage, path in metric_files.items():
        if not path.exists():
            logger.warning(f"[eval_agg] Missing metrics: {stage}")
            continue

        logger.info(f"[eval_agg] Loading {path.name}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        aggregated["stages"][stage] = data

        for key, value in data.items():
            if isinstance(value, (int, float, str, bool)):
                flat_rows.append({
                    "stage": stage,
                    "metric": key,
                    "value": value,
                })

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(aggregated, f, indent=2)

    logger.info(f"[eval_agg] Summary JSON written: {summary_json}")

    if flat_rows:
        with open(summary_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["stage", "metric", "value"]
            )
            writer.writeheader()
            writer.writerows(flat_rows)

        logger.info(f"[eval_agg] Summary CSV written: {summary_csv}")
    else:
        logger.warning("[eval_agg] No scalar metrics for CSV")

    logger.info("[eval_agg] Stage completed")
