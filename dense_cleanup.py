#!/usr/bin/env python3
"""
dense_cleanup.py (Scale-aware)

MARK-2 Dense Point Cloud Cleanup Stage
--------------------------------------

- Reads dense_metrics.json from dense_eval
- Removes outliers and downsamples based on scene scale
- Reduces noise while preserving geometry
- Deterministic & resume-safe
"""

from pathlib import Path
import json
import open3d as o3d
import numpy as np

from utils.paths import ProjectPaths


# ------------------------------------------------------------
# Stage
# ------------------------------------------------------------
def run(run_root: Path, project_root: Path, force: bool, logger):
    run_root = Path(run_root).resolve()
    project_root = Path(project_root).resolve()
    paths = ProjectPaths(project_root)

    logger.info("[dense_cleanup] START")

    dense_dir = paths.dense
    eval_dir = paths.evaluation
    output_dir = dense_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    input_cloud_path = dense_dir / "fused.ply"
    metrics_path = eval_dir / "dense_metrics.json"
    output_cloud_path = output_dir / "fused_cleaned.ply"
    report_path = output_dir / "cleanup_report.json"

    # ------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------
    if not input_cloud_path.exists():
        raise FileNotFoundError(f"[dense_cleanup] Dense cloud not found: {input_cloud_path}")

    if not metrics_path.exists():
        raise FileNotFoundError(f"[dense_cleanup] dense_metrics.json not found: {metrics_path}")

    if output_cloud_path.exists() and not force:
        logger.info("[dense_cleanup] Output exists — skipping")
        return

    # ------------------------------------------------------------
    # Load metrics
    # ------------------------------------------------------------
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    extent = np.array(metrics["bounding_box"]["extent"])
    volume = metrics["bounding_box"]["volume"]
    nn_mean = metrics["nearest_neighbor"]["mean"]

    # Compute scale-aware parameters
    diag = np.linalg.norm(extent)
    voxel_size = diag * 0.003               # small fraction of diagonal
    sor_neighbors = max(20, int(500_000 / metrics["num_points"]))  # more points => fewer neighbors
    sor_std_ratio = 2.0
    radius = max(nn_mean * 3.0, diag * 0.001)   # radius outlier filter
    radius_min_points = 16

    logger.info(f"[dense_cleanup] Scale-aware parameters:")
    logger.info(f"  Diagonal: {diag:.6f}")
    logger.info(f"  Voxel size: {voxel_size:.6f}")
    logger.info(f"  SOR neighbors: {sor_neighbors}, std_ratio: {sor_std_ratio}")
    logger.info(f"  Radius: {radius:.6f}, min points: {radius_min_points}")

    # ------------------------------------------------------------
    # Load point cloud
    # ------------------------------------------------------------
    logger.info(f"[dense_cleanup] Loading: {input_cloud_path}")
    pcd = o3d.io.read_point_cloud(str(input_cloud_path))
    if not pcd.has_points():
        raise RuntimeError("[dense_cleanup] Dense cloud contains no points")

    original_count = len(pcd.points)
    logger.info(f"[dense_cleanup] Original points: {original_count:,}")

    # ------------------------------------------------------------
    # 1. Statistical Outlier Removal
    # ------------------------------------------------------------
    logger.info("[dense_cleanup] Statistical outlier removal")
    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=sor_neighbors,
        std_ratio=sor_std_ratio
    )
    after_sor = len(pcd.points)
    logger.info(f"[dense_cleanup] After SOR: {after_sor:,}")

    # ------------------------------------------------------------
    # 2. Radius Outlier Removal
    # ------------------------------------------------------------
    logger.info("[dense_cleanup] Radius outlier removal")
    pcd, _ = pcd.remove_radius_outlier(
        nb_points=radius_min_points,
        radius=radius
    )
    after_radius = len(pcd.points)
    logger.info(f"[dense_cleanup] After radius filter: {after_radius:,}")

    # ------------------------------------------------------------
    # 3. Voxel Downsampling
    # ------------------------------------------------------------
    logger.info("[dense_cleanup] Voxel downsampling")
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    after_voxel = len(pcd.points)
    logger.info(f"[dense_cleanup] After voxel: {after_voxel:,}")

    # ------------------------------------------------------------
    # Recompute normals (important for Poisson meshing)
    # ------------------------------------------------------------
    normal_radius = diag * 0.01
    max_nn = 30
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius,
            max_nn=max_nn
        )
    )
    pcd.orient_normals_consistent_tangent_plane(k=30)

    # ------------------------------------------------------------
    # Save cleaned cloud
    # ------------------------------------------------------------
    o3d.io.write_point_cloud(str(output_cloud_path), pcd, write_ascii=False)

    reduction_ratio = 1.0 - (after_voxel / original_count)
    report = {
        "input_path": str(input_cloud_path),
        "output_path": str(output_cloud_path),
        "original_points": original_count,
        "after_statistical": after_sor,
        "after_radius": after_radius,
        "after_voxel": after_voxel,
        "reduction_ratio": reduction_ratio,
        "deterministic": True,
        "scale_diag": diag,
        "voxel_size": voxel_size,
        "radius_filter": radius,
        "sor_neighbors": sor_neighbors
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"[dense_cleanup] Reduction ratio: {reduction_ratio:.4f}")
    logger.info("[dense_cleanup] DONE")