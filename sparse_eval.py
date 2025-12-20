#!/usr/bin/env python3
"""
sparse_evaluation.py

MARK-2 Sparse Evaluation (Canonical)
-----------------------------------
- Runs COLMAP model_analyzer
- Writes JSON + CSV metrics
- Runner-managed logger
"""

import json
import csv
import subprocess
import re
import logging
from pathlib import Path

from utils.paths import ProjectPaths
from utils.config import COLMAP_EXE


# -----------------------------
# Logger
# -----------------------------
def make_logger(name: str, log_dir: Path):
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        fh = logging.FileHandler(log_dir / f"{name}.log")
        sh = logging.StreamHandler()
        fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
        fh.setFormatter(fmt)
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)

    return logger


# -----------------------------
# Helpers
# -----------------------------
def run_command(cmd, logger):
    logger.info(" ".join(map(str, cmd)))
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=True,
    )
    logger.info(proc.stdout)
    return proc.stdout


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
    for k, p in patterns.items():
        m = re.search(p, output)
        metrics[k] = float(m.group(1)) if m and "." in m.group(1) else int(m.group(1)) if m else None

    return metrics


# -----------------------------
# Pipeline
# -----------------------------
def run(project_root: Path, force: bool, logger):
    paths = ProjectPaths(project_root)
    paths.ensure_all()

    sparse_root = paths.sparse
    eval_dir = paths.evaluation
    eval_dir.mkdir(parents=True, exist_ok=True)

    models = sorted(
        d for d in sparse_root.iterdir()
        if d.is_dir() and (d / "cameras.bin").exists()
    )

    if not models:
        raise RuntimeError("No valid sparse model found")

    model = models[0]

    json_out = eval_dir / "sparse_metrics.json"
    csv_out = eval_dir / "sparse_metrics.csv"

    if json_out.exists() and not force:
        logger.info("Sparse metrics already exist — skipping")
        return

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

    json_out.write_text(json.dumps(metrics, indent=2))
    logger.info(f"Wrote JSON: {json_out}")

    with open(csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metrics.keys())
        writer.writeheader()
        writer.writerow(metrics)

    logger.info(f"Wrote CSV: {csv_out}")
    logger.info("Sparse evaluation complete")
