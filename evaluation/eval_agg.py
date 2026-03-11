#!/usr/bin/env python3
"""
eval_agg.py

MARK-2 Evaluation Aggregation Stage
-----------------------------------

Collects all evaluation outputs and produces a unified report.

Inputs
------
runs/<run_id>/evaluation/
    sparse_metrics.json
    dense_metrics.json
    mesh_metrics.json

Outputs
-------
summary.json
summary.csv
"""

import json
import csv
from pathlib import Path
from datetime import datetime


# --------------------------------------------------------
# Helpers
# --------------------------------------------------------

def load_json_safe(path: Path, logger):

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    except Exception as e:
        logger.warning(f"[eval_agg] failed loading {path.name}: {e}")
        return None


def flatten_metrics(stage_name, metrics):

    rows = []

    for k, v in metrics.items():

        if isinstance(v, (int, float, str, bool)):

            rows.append({
                "stage": stage_name,
                "metric": k,
                "value": v
            })

    return rows


# --------------------------------------------------------
# Pipeline Stage
# --------------------------------------------------------

def run(paths, tools, config, logger):

    logger.info("[eval_agg] stage start")

    eval_dir: Path = paths.evaluation

    if not eval_dir.exists():
        raise RuntimeError("evaluation directory missing")

    summary_json = eval_dir / "summary.json"
    summary_csv = eval_dir / "summary.csv"

    metric_files = {
        "sparse": eval_dir / "sparse_metrics.json",
        "dense": eval_dir / "dense_metrics.json",
        "mesh": eval_dir / "mesh_metrics.json",
    }

    aggregated = {
        "metadata": {
            "project": config["project"]["name"],
            "run_id": paths.run_id,
            "dataset_path": config["project"]["dataset_path"],
            "run_root": str(paths.root),
            "timestamp": datetime.utcnow().isoformat()
        },
        "stages": {}
    }

    csv_rows = []

    # ----------------------------------------------------
    # Load Metrics
    # ----------------------------------------------------

    for stage, metric_path in metric_files.items():

        if not metric_path.exists():

            logger.warning(f"[eval_agg] missing metrics: {metric_path.name}")
            continue

        logger.info(f"[eval_agg] loading {metric_path.name}")

        metrics = load_json_safe(metric_path, logger)

        if metrics is None:
            continue

        aggregated["stages"][stage] = metrics

        csv_rows.extend(flatten_metrics(stage, metrics))

    # ----------------------------------------------------
    # Write JSON
    # ----------------------------------------------------

    with open(summary_json, "w", encoding="utf-8") as f:

        json.dump(
            aggregated,
            f,
            indent=2,
            sort_keys=True
        )

    logger.info(f"[eval_agg] wrote {summary_json}")

    # ----------------------------------------------------
    # Write CSV
    # ----------------------------------------------------

    if csv_rows:

        with open(summary_csv, "w", newline="", encoding="utf-8") as f:

            writer = csv.DictWriter(
                f,
                fieldnames=["stage", "metric", "value"]
            )

            writer.writeheader()
            writer.writerows(csv_rows)

        logger.info(f"[eval_agg] wrote {summary_csv}")

    else:

        logger.warning("[eval_agg] no scalar metrics found")

    logger.info("[eval_agg] stage completed")