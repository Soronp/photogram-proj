#!/usr/bin/env python3

"""
Dense Evaluation Stage

Evaluates dense point cloud geometry statistics.
"""

import json
import numpy as np
import open3d as o3d


def run(paths, logger, tools, config):

    logger.info("[dense_eval] starting")

    dense_cleaned = paths.dense / "fused_cleaned.ply"
    dense_raw = paths.dense / "fused.ply"

    dense_ply = dense_cleaned if dense_cleaned.exists() else dense_raw

    if not dense_ply.exists():
        raise RuntimeError("dense point cloud missing")

    eval_dir = paths.evaluation

    eval_dir.mkdir(parents=True, exist_ok=True)

    out_path = eval_dir / "dense_metrics.json"

    if out_path.exists():

        logger.info("[dense_eval] metrics exist — skipping")

        return

    # -------------------------------------------------
    # Load point cloud
    # -------------------------------------------------

    pcd = o3d.io.read_point_cloud(str(dense_ply))

    if not pcd.has_points():
        raise RuntimeError("dense point cloud empty")

    points = np.asarray(pcd.points)

    num_points = int(points.shape[0])

    logger.info(f"[dense_eval] loaded {num_points:,} points")

    # -------------------------------------------------
    # Bounding box
    # -------------------------------------------------

    aabb = pcd.get_axis_aligned_bounding_box()

    extent = aabb.get_extent()

    volume = float(np.prod(extent))

    # -------------------------------------------------
    # Nearest neighbor statistics
    # -------------------------------------------------

    logger.info("[dense_eval] computing NN statistics")

    kdtree = o3d.geometry.KDTreeFlann(pcd)

    nn_distances = []

    stride = max(1, num_points // 200_000)

    for i in range(0, num_points, stride):

        _, _, dists = kdtree.search_knn_vector_3d(
            pcd.points[i],
            2
        )

        if len(dists) > 1:
            nn_distances.append(np.sqrt(dists[1]))

    nn = np.array(nn_distances) if nn_distances else np.array([0.0])

    # -------------------------------------------------
    # Metrics
    # -------------------------------------------------

    metrics = {

        "num_points": num_points,

        "bounding_box": {
            "extent": extent.tolist(),
            "volume": volume
        },

        "point_density":
            float(num_points / volume) if volume > 0 else 0.0,

        "nearest_neighbor": {
            "mean": float(np.mean(nn)),
            "median": float(np.median(nn)),
            "std": float(np.std(nn)),
        },

        "notes":
            "geometry-only dense evaluation"
    }

    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"[dense_eval] metrics written → {out_path}")

    logger.info("[dense_eval] completed")