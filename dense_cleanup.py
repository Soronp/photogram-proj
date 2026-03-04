#!/usr/bin/env python3
"""
MARK-2 Dense Cleanup (Minimal Geometry Loss)
"""

from pathlib import Path
import json
import open3d as o3d
import numpy as np
from utils.paths import ProjectPaths


def run(run_root: Path, project_root: Path, force: bool, logger):
    paths = ProjectPaths(project_root)

    dense_dir = paths.dense
    eval_dir = paths.evaluation

    input_cloud = dense_dir / "fused.ply"
    metrics_path = eval_dir / "dense_metrics.json"
    output_cloud = dense_dir / "fused_cleaned.ply"

    if output_cloud.exists() and not force:
        logger.info("[dense_cleanup] Output exists — skipping")
        return

    if not input_cloud.exists():
        raise FileNotFoundError("Missing fused.ply")

    pcd = o3d.io.read_point_cloud(str(input_cloud))
    pcd.remove_non_finite_points()

    original = len(pcd.points)

    # Light statistical removal only
    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=30,
        std_ratio=2.5
    )

    cleaned = len(pcd.points)

    o3d.io.write_point_cloud(str(output_cloud), pcd, write_ascii=False)

    logger.info(f"[dense_cleanup] Points: {original:,} → {cleaned:,}")
    logger.info("[dense_cleanup] DONE")