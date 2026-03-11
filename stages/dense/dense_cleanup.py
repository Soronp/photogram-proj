#!/usr/bin/env python3
"""
dense_cleanup.py

MARK-2 Pipeline Stage
Dense point cloud cleanup.

Input
-----
dense/fused.ply

Output
------
dense/fused_cleaned.ply
"""

from pathlib import Path
import numpy as np
import open3d as o3d


# --------------------------------------------------
# Locate dense cloud
# --------------------------------------------------

def locate_dense_cloud(paths):

    candidates = [

        paths.dense / "fused.ply",
        paths.openmvs / "fused.ply"
    ]

    for c in candidates:

        if c.exists() and c.stat().st_size > 10000:
            return c

    raise RuntimeError("Dense cloud not found (fused.ply missing)")


# --------------------------------------------------
# Validate point cloud
# --------------------------------------------------

def validate_cloud(pcd):

    if len(pcd.points) == 0:
        raise RuntimeError("Point cloud empty")

    if not pcd.has_points():
        raise RuntimeError("Invalid point cloud")

    return len(pcd.points)


# --------------------------------------------------
# Compute scene scale
# --------------------------------------------------

def compute_scene_scale(pcd):

    bbox = pcd.get_axis_aligned_bounding_box()

    extent = bbox.get_extent()

    diag = np.linalg.norm(extent)

    return diag


# --------------------------------------------------
# Remove statistical outliers
# --------------------------------------------------

def remove_noise(pcd, logger):

    nb_neighbors = 30
    std_ratio = 2.5

    filtered, _ = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio
    )

    logger.info(
        f"[dense_cleanup] after noise filter → {len(filtered.points):,}"
    )

    return filtered


# --------------------------------------------------
# Adaptive downsampling
# --------------------------------------------------

def adaptive_downsample(pcd, scene_diag, logger):

    count = len(pcd.points)

    if count < 5_000_000:

        logger.info("[dense_cleanup] downsample skipped (small cloud)")

        return pcd

    if scene_diag < 3.0:
        voxel = scene_diag * 0.004
    else:
        voxel = scene_diag * 0.008

    pcd_ds = pcd.voxel_down_sample(voxel)

    if len(pcd_ds.points) < 50_000:

        logger.info("[dense_cleanup] downsample rejected (too aggressive)")

        return pcd

    logger.info(
        f"[dense_cleanup] voxel downsample {voxel:.6f} → {len(pcd_ds.points):,}"
    )

    return pcd_ds


# --------------------------------------------------
# Stage entry
# --------------------------------------------------

def run(paths, logger, tools, config):

    logger.info("[dense_cleanup] starting")

    dense_dir = paths.dense
    dense_dir.mkdir(parents=True, exist_ok=True)

    output_cloud = dense_dir / "fused_cleaned.ply"

    if output_cloud.exists():

        logger.info("[dense_cleanup] output exists — skipping")

        return

    # --------------------------------------------------
    # Locate input
    # --------------------------------------------------

    input_cloud = locate_dense_cloud(paths)

    logger.info(f"[dense_cleanup] input cloud → {input_cloud}")

    # --------------------------------------------------
    # Load cloud
    # --------------------------------------------------

    pcd = o3d.io.read_point_cloud(str(input_cloud))

    pcd.remove_non_finite_points()

    original_count = validate_cloud(pcd)

    logger.info(f"[dense_cleanup] loaded {original_count:,} points")

    # --------------------------------------------------
    # Scene scale
    # --------------------------------------------------

    scene_diag = compute_scene_scale(pcd)

    logger.info(f"[dense_cleanup] scene diagonal → {scene_diag:.4f}")

    # --------------------------------------------------
    # Noise removal
    # --------------------------------------------------

    pcd = remove_noise(pcd, logger)

    # --------------------------------------------------
    # Downsampling
    # --------------------------------------------------

    pcd = adaptive_downsample(pcd, scene_diag, logger)

    # --------------------------------------------------
    # Save result
    # --------------------------------------------------

    o3d.io.write_point_cloud(
        str(output_cloud),
        pcd,
        write_ascii=False
    )

    logger.info(
        f"[dense_cleanup] final points → {len(pcd.points):,}"
    )

    logger.info(f"[dense_cleanup] output → {output_cloud}")

    logger.info("[dense_cleanup] completed")