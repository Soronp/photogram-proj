#!/usr/bin/env python3
"""
sparse_evaluation.py

MARK-2 Sparse Evaluation Stage
------------------------------
Responsibilities:
- Run COLMAP model_analyzer
- Parse reconstruction metrics
- Emit JSON and CSV reports
"""

import json
import csv
import subprocess
import re
from pathlib import Path

from utils.paths import ProjectPaths
from utils.config import COLMAP_EXE


# --------------------------------------------------
# Command Runner
# --------------------------------------------------
def run_command(cmd, logger):
    logger.info("[sparse_eval] " + " ".join(map(str, cmd)))

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    if proc.stdout.strip():
        logger.info(proc.stdout)

    if proc.returncode != 0:
        raise RuntimeError("[sparse_eval] model_analyzer failed")

    return proc.stdout


# --------------------------------------------------
# Output Parsing
# --------------------------------------------------
def parse_metrics(output: str) -> dict:
    patterns = {
        "num_images": r"Registered images:\s+(\d+)",
        "num_points": r"Points:\s+(\d+)",
        "num_observations": r"Observations:\s+(\d+)",
        "mean_track_length": r"Mean track length:\s+([\d.]+)",
        "mean_observations_per_image": r"Mean observations per image:\s+([\d.]+)",
        "mean_reprojection_error": r"Mean reprojection error:\s+([\d.]+)",
    }

    metrics = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, output)
        if not match:
            metrics[key] = None
            continue

        value = match.group(1)
        metrics[key] = float(value) if "." in value else int(value)

    return metrics


# --------------------------------------------------
# Pipeline Stage
# --------------------------------------------------
def run(run_root: Path, project_root: Path, force: bool, logger):
    paths = ProjectPaths(project_root)
    paths.ensure_all()

    sparse_root = paths.sparse
    eval_dir = paths.evaluation
    eval_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # Select canonical sparse model
    # --------------------------------------------------
    models = sorted(
        d for d in sparse_root.iterdir()
        if d.is_dir() and (d / "cameras.bin").exists()
    )

    if not models:
        raise RuntimeError("[sparse_eval] No valid sparse model found")

    model = models[0]
    logger.info(f"[sparse_eval] Evaluating model: {model.name}")

    json_out = eval_dir / "sparse_metrics.json"
    csv_out = eval_dir / "sparse_metrics.csv"

    if json_out.exists() and not force:
        logger.info("[sparse_eval] Metrics already exist — skipping")
        return

    # --------------------------------------------------
    # Run COLMAP Analyzer
    # --------------------------------------------------
    output = run_command(
        [COLMAP_EXE, "model_analyzer", "--path", model],
        logger,
    )

    metrics = parse_metrics(output)
    metrics.update({
        "model_path": str(model),
        "stage": "sparse",
        "openmvs_ready": True,
    })

    # --------------------------------------------------
    # Write Outputs
    # --------------------------------------------------
    json_out.write_text(json.dumps(metrics, indent=2))
    logger.info(f"[sparse_eval] Wrote JSON: {json_out}")

    with open(csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metrics.keys())
        writer.writeheader()
        writer.writerow(metrics)

    logger.info(f"[sparse_eval] Wrote CSV: {csv_out}")
    logger.info("[sparse_eval] Sparse evaluation complete")
