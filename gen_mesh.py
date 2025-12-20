#!/usr/bin/env python3
"""
gen_mesh.py

MARK-2 Mesh Generation Stage
---------------------------
- Poisson surface reconstruction
- Deterministic, resume-safe
"""

import shutil
from pathlib import Path
import numpy as np
import open3d as o3d

from utils.paths import ProjectPaths
from utils.config import load_config


# --------------------------------------------------
# PIPELINE STAGE
# --------------------------------------------------
def run(project_root: Path, force: bool, logger):
    paths = ProjectPaths(project_root)
    config = load_config(project_root)

    dense_ply = paths.dense / "fused.ply"
    mesh_dir = paths.mesh
    mesh_out = mesh_dir / "mesh_raw.ply"

    mesh_cfg = config.get("mesh", {})

    logger.info("[mesh] Stage started")
    logger.info(f"[mesh] Input : {dense_ply}")
    logger.info(f"[mesh] Output: {mesh_out}")

    if not dense_ply.exists():
        raise FileNotFoundError(f"Dense point cloud not found: {dense_ply}")

    if mesh_out.exists() and not force:
        logger.info("[mesh] Mesh exists — skipping")
        return

    if mesh_dir.exists() and force:
        shutil.rmtree(mesh_dir)

    mesh_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # Load point cloud
    # --------------------------------------------------
    pcd = o3d.io.read_point_cloud(str(dense_ply))
    if not pcd.has_points():
        raise RuntimeError("Dense point cloud is empty")

    pcd.remove_non_finite_points()
    logger.info(f"[mesh] Points after cleanup: {len(pcd.points):,}")

    # --------------------------------------------------
    # Normal estimation
    # --------------------------------------------------
    bbox = pcd.get_axis_aligned_bounding_box()
    diag = np.linalg.norm(bbox.get_extent())

    radius = diag * 0.01
    max_nn = 30
    logger.info(f"[mesh] Estimating normals (r={radius:.4f}, nn={max_nn})")

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius,
            max_nn=max_nn,
        )
    )
    pcd.normalize_normals()

    # --------------------------------------------------
    # Poisson reconstruction
    # --------------------------------------------------
    depth = int(mesh_cfg.get("poisson_depth", 10))
    logger.info(f"[mesh] Poisson reconstruction (depth={depth})")

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth
    )

    if not mesh.has_triangles():
        raise RuntimeError("Poisson reconstruction produced no triangles")

    logger.info(
        f"[mesh] Raw mesh: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles"
    )

    # --------------------------------------------------
    # Optional density trimming
    # --------------------------------------------------
    q = mesh_cfg.get("density_trim_quantile")
    if q is not None and 0.0 < float(q) < 1.0:
        densities = np.asarray(densities)
        cutoff = np.quantile(densities, float(q))
        mask = densities < cutoff

        removed = int(mask.sum())
        mesh.remove_vertices_by_mask(mask)

        logger.info(
            f"[mesh] Density trim removed {removed:,} vertices "
            f"(remaining {len(mesh.vertices):,})"
        )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------
    o3d.io.write_triangle_mesh(str(mesh_out), mesh, write_ascii=False)
    logger.info(f"[mesh] Mesh written: {mesh_out}")
    logger.info("[mesh] Stage completed")
