#!/usr/bin/env python3
"""
gen_mesh.py

MARK-2 Mesh Generation Stage (Canonical)
---------------------------------------

- Uses ProjectPaths (single filesystem authority)
- Consumes cleaned dense cloud if available
- Falls back to raw fused.ply
- Deterministic
- Resume-safe
"""

from pathlib import Path
import shutil
import numpy as np
import open3d as o3d

from utils.paths import ProjectPaths
from utils.config import load_config


# --------------------------------------------------
# PIPELINE STAGE
# --------------------------------------------------
def run(run_root: Path, project_root: Path, force: bool, logger):

    run_root = Path(run_root).resolve()
    project_root = Path(project_root).resolve()

    paths = ProjectPaths(project_root)
    config = load_config(run_root)

    mesh_cfg = config.get("mesh", {})

    logger.info("[mesh] START")

    # --------------------------------------------------
    # Canonical paths
    # --------------------------------------------------

    dense_dir = paths.dense                     # root/openmvs/dense
    cleaned_dir = dense_dir / "cleaned"
    mesh_dir = paths.mesh                       # root/mesh

    raw_ply = dense_dir / "fused.ply"
    cleaned_ply = cleaned_dir / "dense_cleaned.ply"

    mesh_out = mesh_dir / "mesh_raw.ply"

    # --------------------------------------------------
    # Select input cloud
    # --------------------------------------------------

    if cleaned_ply.exists():
        input_cloud = cleaned_ply
        logger.info("[mesh] Using cleaned dense cloud")
    elif raw_ply.exists():
        input_cloud = raw_ply
        logger.warning("[mesh] Cleanup missing — using raw fused.ply")
    else:
        raise FileNotFoundError(
            f"[mesh] Dense cloud not found:\n"
            f"  - {cleaned_ply}\n"
            f"  - {raw_ply}"
        )

    logger.info(f"[mesh] Input : {input_cloud}")
    logger.info(f"[mesh] Output: {mesh_out}")

    # --------------------------------------------------
    # Skip logic
    # --------------------------------------------------

    if mesh_out.exists() and not force:
        logger.info("[mesh] Output exists — skipping")
        return

    if mesh_dir.exists() and force:
        shutil.rmtree(mesh_dir)

    mesh_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # Load point cloud
    # --------------------------------------------------

    pcd = o3d.io.read_point_cloud(str(input_cloud))

    if not pcd.has_points():
        raise RuntimeError("[mesh] Input point cloud is empty")

    pcd.remove_non_finite_points()

    logger.info(f"[mesh] Points: {len(pcd.points):,}")

    # --------------------------------------------------
    # Normal estimation
    # --------------------------------------------------

    bbox = pcd.get_axis_aligned_bounding_box()
    diag = np.linalg.norm(bbox.get_extent())

    if diag <= 0:
        raise RuntimeError("[mesh] Degenerate geometry (invalid bounding box)")

    radius = diag * 0.01
    max_nn = 30

    logger.info(
        f"[mesh] Estimating normals (radius={radius:.6f}, max_nn={max_nn})"
    )

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
        pcd,
        depth=depth,
    )

    if not mesh.has_triangles():
        raise RuntimeError("[mesh] Poisson reconstruction failed")

    logger.info(
        f"[mesh] Raw mesh: "
        f"{len(mesh.vertices):,} vertices | "
        f"{len(mesh.triangles):,} triangles"
    )

    # --------------------------------------------------
    # Density trimming (optional)
    # --------------------------------------------------

    q = mesh_cfg.get("density_trim_quantile")

    if q is not None:
        q = float(q)
        if 0.0 < q < 1.0:
            densities = np.asarray(densities)
            cutoff = np.quantile(densities, q)
            mask = densities < cutoff
            removed = int(mask.sum())

            mesh.remove_vertices_by_mask(mask)

            logger.info(
                f"[mesh] Density trim removed {removed:,} vertices"
            )

    # --------------------------------------------------
    # Final cleanup
    # --------------------------------------------------

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    logger.info(
        f"[mesh] Final mesh: "
        f"{len(mesh.vertices):,} vertices | "
        f"{len(mesh.triangles):,} triangles"
    )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    o3d.io.write_triangle_mesh(
        str(mesh_out),
        mesh,
        write_ascii=False,
    )

    logger.info(f"[mesh] Mesh written: {mesh_out}")
    logger.info("[mesh] DONE")