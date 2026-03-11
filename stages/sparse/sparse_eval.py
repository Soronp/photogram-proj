#!/usr/bin/env python3
"""
sparse_eval.py

Stage — Sparse reconstruction evaluation

Uses COLMAP model_analyzer to extract reconstruction statistics.
"""

import json
import csv
import re
import subprocess
from pathlib import Path


# --------------------------------------------------
# Output Parsing
# --------------------------------------------------

def parse_analyzer_output(text: str):

    patterns = {
        "registered_images": r"Registered images:\s+(\d+)",
        "points3D": r"Points:\s+(\d+)",
        "observations": r"Observations:\s+(\d+)",
        "mean_track_length": r"Mean track length:\s+([\d.]+)",
        "mean_observations_per_image": r"Mean observations per image:\s+([\d.]+)",
        "mean_reprojection_error": r"Mean reprojection error:\s+([\d.]+)",
    }

    metrics = {}

    for key, pattern in patterns.items():

        match = re.search(pattern, text)

        if not match:
            metrics[key] = None
            continue

        value = match.group(1)

        try:
            metrics[key] = float(value) if "." in value else int(value)
        except Exception:
            metrics[key] = None

    return metrics


# --------------------------------------------------
# Model discovery
# --------------------------------------------------

def find_sparse_model(sparse_root: Path):

    models = [
        d for d in sparse_root.iterdir()
        if d.is_dir() and (d / "cameras.bin").exists()
    ]

    if not models:
        raise RuntimeError("No sparse reconstruction found")

    return sorted(models)[0]


# --------------------------------------------------
# Run analyzer safely
# --------------------------------------------------

def run_analyzer(model_path: Path):

    cmd = [
        "colmap",
        "model_analyzer",
        "--path",
        str(model_path)
    ]

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    return proc.stdout


# --------------------------------------------------
# Stage execution
# --------------------------------------------------

def run(paths, logger, tools, config):

    logger.info("[sparse_eval] starting")

    sparse_root = paths.sparse
    eval_dir = paths.evaluation

    eval_dir.mkdir(parents=True, exist_ok=True)

    model = find_sparse_model(sparse_root)

    logger.info(f"[sparse_eval] model: {model.name}")

    # -----------------------------------------
    # Run analyzer
    # -----------------------------------------

    output = run_analyzer(model)

    if not output or not output.strip():
        raise RuntimeError("COLMAP model_analyzer returned no output")

    logger.info("[sparse_eval] analyzer output captured")

    metrics = parse_analyzer_output(output)

    metrics.update({
        "stage": "sparse",
        "model_path": str(model),
        "dense_ready": True,
    })

    # -----------------------------------------
    # Write reports
    # -----------------------------------------

    json_path = eval_dir / "sparse_metrics.json"
    csv_path = eval_dir / "sparse_metrics.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=metrics.keys())
        writer.writeheader()
        writer.writerow(metrics)

    logger.info(f"[sparse_eval] metrics written → {json_path.name}")
    logger.info("[sparse_eval] completed")