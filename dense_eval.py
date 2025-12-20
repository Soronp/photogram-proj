#!/usr/bin/env python3
"""
dense_evaluation.py

MARK-2 Dense Reconstruction Evaluation Stage
--------------------------------------------
Geometry-only dense cloud evaluation.
Runner-managed, deterministic, resume-safe.
"""

import json
import numpy as np
from pathlib import Path
import open3d as o3d

from utils.paths import ProjectPaths
from utils.config import load_config


# --------------------------------------------------
# PIPELINE STAGE
# --------------------------------------------------
def run(project_root: Path, force: bool, logger):
    paths = ProjectPaths(project_root)
    _ = load_config(project_root)  # reserved for thresholds

    dense_ply = paths.dense / "fused.ply"
    eval_dir = paths.evaluation
    eval_dir.mkdir(parents=True, exist_ok=True)

    out_path = eval_dir / "dense_metrics.json"

    logger.info("[dense_eval] Stage started")
    logger.info(f"[dense_eval] Input: {dense_ply}")

    if out_path.exists() and not force:
        logger.info("[dense_eval] Metrics already exist — skipping")
        return

    if not dense_ply.exists():
        raise FileNotFoundError(f"Dense point cloud not found: {dense_ply}")

    # --------------------------------------------------
    # Load point cloud
    # --------------------------------------------------
    pcd = o3d.io.read_point_cloud(str(dense_ply))
    if not pcd.has_points():
        raise RuntimeError("Dense point cloud is empty")

    points = np.asarray(pcd.points)
    num_points = int(points.shape[0])

    logger.info(f"[dense_eval] Loaded {num_points:,} points")

    # --------------------------------------------------
    # Bounding box
    # --------------------------------------------------
    aabb = pcd.get_axis_aligned_bounding_box()
    extent = aabb.get_extent()
    volume = float(np.prod(extent))

    # --------------------------------------------------
    # Nearest neighbor statistics (deterministic subsample)
    # --------------------------------------------------
    logger.info("[dense_eval] Computing NN statistics")

    kdtree = o3d.geometry.KDTreeFlann(pcd)
    nn_distances = []

    stride = max(1, num_points // 200_000)

    for i in range(0, num_points, stride):
        _, _, dists = kdtree.search_knn_vector_3d(pcd.points[i], 2)
        if len(dists) > 1:
            nn_distances.append(np.sqrt(dists[1]))

    nn = np.array(nn_distances)

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------
    metrics = {
        "num_points": num_points,
        "bounding_box": {
            "extent": extent.tolist(),
            "volume": volume,
        },
        "point_density": float(num_points / volume) if volume > 0 else 0.0,
        "nearest_neighbor": {
            "mean": float(np.mean(nn)),
            "median": float(np.median(nn)),
            "std": float(np.std(nn)),
        },
        "notes": "Geometry-only dense evaluation (no ground truth)",
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"[dense_eval] Metrics written: {out_path}")
    logger.info("[dense_eval] Stage completed")
