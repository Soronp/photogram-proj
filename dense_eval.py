#!/usr/bin/env python3
"""
dense_evaluation.py

MARK-2 Dense Reconstruction Evaluation Stage
-------------------------------------------
Evaluates the quality of the dense point cloud produced by COLMAP.

Metrics (geometry-only, deterministic):
- Number of points
- Bounding box size / volume
- Point density (points per cubic unit)
- Nearest-neighbor distance statistics (mean / median / std)
- Basic noise proxy (std of NN distances)

Reads:
- dense/fused.ply

Writes:
- evaluation/dense_metrics.json
- logs/dense_evaluation.log
"""

import json
import numpy as np
from pathlib import Path
import open3d as o3d
import argparse

from utils.logger import get_logger
from utils.paths import ProjectPaths
from utils.config import load_config

# ------------------------------------------------------------------
# Dense evaluation
# ------------------------------------------------------------------

def run_dense_evaluation(project_root: Path, force: bool = False):
    """
    Evaluate dense reconstruction quality.
    Accepts 'force' for pipeline runner compatibility.
    """
    paths = ProjectPaths(project_root)
    _ = load_config(project_root)  # reserved for future thresholds
    logger = get_logger("dense_evaluation", project_root)

    dense_ply = paths.dense / "fused.ply"
    eval_dir = paths.evaluation
    eval_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting dense reconstruction evaluation")
    logger.info(f"Dense input: {dense_ply}")

    if not dense_ply.exists():
        raise FileNotFoundError(f"Dense point cloud not found: {dense_ply}")

    # --------------------------------------------------
    # Load dense point cloud
    # --------------------------------------------------
    pcd = o3d.io.read_point_cloud(str(dense_ply))
    if not pcd.has_points():
        raise RuntimeError("Dense point cloud is empty")

    points = np.asarray(pcd.points)
    num_points = points.shape[0]
    logger.info(f"Loaded dense cloud with {num_points:,} points")

    # --------------------------------------------------
    # Bounding box + volume
    # --------------------------------------------------
    aabb = pcd.get_axis_aligned_bounding_box()
    extent = aabb.get_extent()
    volume = float(np.prod(extent))

    # --------------------------------------------------
    # Nearest neighbor statistics
    # --------------------------------------------------
    logger.info("Computing nearest-neighbor statistics")
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    nn_distances = []

    stride = max(1, num_points // 200_000)  # deterministic subsampling

    for i in range(0, num_points, stride):
        _, idx, dists = kdtree.search_knn_vector_3d(pcd.points[i], 2)
        if len(dists) > 1:
            nn_distances.append(np.sqrt(dists[1]))

    nn_distances = np.array(nn_distances)
    nn_mean = float(np.mean(nn_distances))
    nn_median = float(np.median(nn_distances))
    nn_std = float(np.std(nn_distances))

    # --------------------------------------------------
    # Density estimate
    # --------------------------------------------------
    density = float(num_points / volume) if volume > 0 else 0.0

    # --------------------------------------------------
    # Save metrics
    # --------------------------------------------------
    metrics = {
        "num_points": int(num_points),
        "bounding_box": {
            "extent": extent.tolist(),
            "volume": volume
        },
        "point_density": density,
        "nearest_neighbor": {
            "mean": nn_mean,
            "median": nn_median,
            "std": nn_std
        },
        "notes": "Geometry-only dense evaluation (no ground truth)"
    }

    out_path = eval_dir / "dense_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Dense evaluation completed successfully")
    logger.info(f"Metrics written to: {out_path}")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MARK-2 Dense Reconstruction Evaluation")
    parser.add_argument("project_root", type=Path, help="Project root directory")
    parser.add_argument("--force", action="store_true", help="Force re-evaluation")
    args = parser.parse_args()

    run_dense_evaluation(args.project_root, force=args.force)


if __name__ == "__main__":
    main()
