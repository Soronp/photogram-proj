#!/usr/bin/env python3
"""
dense_cleanup.py

MARK-2 Dense Point Cloud Cleanup Stage
--------------------------------------

Position:
    Between dense_eval and mesh

Purpose:
    - Remove noise and outliers
    - Reduce redundancy
    - Improve downstream meshing stability
    - Deterministic
    - Resume-safe
    - Fully compliant with ProjectPaths authority
"""

from pathlib import Path
import json
import open3d as o3d

from utils.paths import ProjectPaths


# ------------------------------------------------------------
# Stage
# ------------------------------------------------------------

def run(run_root: Path, project_root: Path, force: bool, logger):

    run_root = Path(run_root).resolve()
    project_root = Path(project_root).resolve()

    paths = ProjectPaths(project_root)

    logger.info("[dense_cleanup] START")

    # ------------------------------------------------------------
    # Canonical locations (DO NOT HARDCODE)
    # ------------------------------------------------------------

    dense_dir = paths.dense                  # root/openmvs/dense
    eval_dir = paths.evaluation              # root/evaluation
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
        raise FileNotFoundError(
            f"[dense_cleanup] Dense cloud not found: {input_cloud_path}"
        )

    if not metrics_path.exists():
        logger.warning(
            "[dense_cleanup] dense_eval metrics not found — continuing without them"
        )

    if output_cloud_path.exists() and not force:
        logger.info("[dense_cleanup] Output exists — skipping")
        return

    # ------------------------------------------------------------
    # Load cloud
    # ------------------------------------------------------------

    logger.info(f"[dense_cleanup] Loading: {input_cloud_path}")
    pcd = o3d.io.read_point_cloud(str(input_cloud_path))

    if not pcd.has_points():
        raise RuntimeError("Dense cloud contains no points")

    original_count = len(pcd.points)
    logger.info(f"[dense_cleanup] Original points: {original_count:,}")

    # ------------------------------------------------------------
    # 1. Statistical Outlier Removal
    # ------------------------------------------------------------

    logger.info("[dense_cleanup] Statistical outlier removal")

    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=20,
        std_ratio=2.0
    )

    after_sor = len(pcd.points)
    logger.info(f"[dense_cleanup] After SOR: {after_sor:,}")

    # ------------------------------------------------------------
    # 2. Radius Outlier Removal
    # ------------------------------------------------------------

    logger.info("[dense_cleanup] Radius outlier removal")

    pcd, _ = pcd.remove_radius_outlier(
        nb_points=16,
        radius=0.02
    )

    after_radius = len(pcd.points)
    logger.info(f"[dense_cleanup] After radius filter: {after_radius:,}")

    # ------------------------------------------------------------
    # 3. Voxel Downsampling
    # ------------------------------------------------------------

    logger.info("[dense_cleanup] Voxel downsampling")

    pcd = pcd.voxel_down_sample(voxel_size=0.003)

    after_voxel = len(pcd.points)
    logger.info(f"[dense_cleanup] After voxel: {after_voxel:,}")

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
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"[dense_cleanup] Reduction ratio: {reduction_ratio:.4f}")
    logger.info("[dense_cleanup] DONE")