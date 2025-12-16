#!/usr/bin/env python3
"""
gen_mesh.py

MARK-2 Mesh Generation Stage
---------------------------
- Generates surface mesh from dense point cloud
- Algorithm: Poisson Reconstruction (deterministic)
- Input : dense/fused.ply
- Output: mesh/mesh_raw.ply
- Restart-safe, config-driven, logged
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import open3d as o3d

from utils.logger import get_logger
from utils.paths import ProjectPaths
from utils.config import load_config


# ------------------------------------------------------------------
# Mesh generation pipeline
# ------------------------------------------------------------------
def run_mesh_generation(project_root: Path, force: bool):
    project_root = project_root.resolve()
    paths = ProjectPaths(project_root)
    config = load_config(project_root)
    logger = get_logger("gen_mesh", project_root)

    dense_ply = paths.dense / "fused.ply"
    mesh_dir = paths.mesh
    mesh_out = mesh_dir / "mesh_raw.ply"

    mesh_cfg = config.get("mesh", {})

    logger.info("=== MARK-2 Mesh Generation ===")
    logger.info(f"Dense input : {dense_ply}")
    logger.info(f"Mesh output : {mesh_out}")

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------
    if not dense_ply.exists():
        raise FileNotFoundError(f"Dense point cloud not found: {dense_ply}")

    if mesh_out.exists() and not force:
        logger.info("Mesh already exists — skipping (use --force to regenerate)")
        return

    if mesh_dir.exists() and force:
        logger.warning("Removing existing mesh directory (--force)")
        shutil.rmtree(mesh_dir)

    mesh_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # Load dense point cloud
    # --------------------------------------------------
    logger.info("Loading dense point cloud")
    pcd = o3d.io.read_point_cloud(str(dense_ply))

    if not pcd.has_points():
        raise RuntimeError("Dense point cloud contains no points")

    logger.info(f"Loaded {len(pcd.points):,} points")

    pcd.remove_non_finite_points()
    logger.info(f"After cleanup: {len(pcd.points):,} points")

    # --------------------------------------------------
    # Normal estimation (scale-aware)
    # --------------------------------------------------
    bbox = pcd.get_axis_aligned_bounding_box()
    diag = np.linalg.norm(bbox.get_extent())

    normal_radius = diag * 0.01
    normal_max_nn = 30

    logger.info(
        f"Estimating normals "
        f"(radius={normal_radius:.4f}, max_nn={normal_max_nn})"
    )

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius,
            max_nn=normal_max_nn
        )
    )
    pcd.normalize_normals()

    # --------------------------------------------------
    # Poisson reconstruction
    # --------------------------------------------------
    poisson_depth = int(mesh_cfg.get("poisson_depth", 10))
    logger.info(f"Running Poisson reconstruction (depth={poisson_depth})")

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=poisson_depth
    )

    if not mesh.has_triangles():
        raise RuntimeError("Poisson reconstruction failed — no triangles generated")

    logger.info(
        f"Initial mesh: "
        f"{len(mesh.vertices):,} vertices, "
        f"{len(mesh.triangles):,} triangles"
    )

    # --------------------------------------------------
    # Density-based trimming (optional)
    # --------------------------------------------------
    density_q = mesh_cfg.get("density_trim_quantile")

    if density_q is not None:
        q = float(density_q)
        if 0.0 < q < 1.0:
            logger.info(f"Applying density trimming (quantile={q})")
            densities = np.asarray(densities)
            cutoff = np.quantile(densities, q)
            remove_mask = densities < cutoff
            removed = int(np.sum(remove_mask))

            mesh.remove_vertices_by_mask(remove_mask)

            logger.info(
                f"Removed {removed:,} vertices "
                f"(remaining: {len(mesh.vertices):,})"
            )
        else:
            logger.warning("mesh.density_trim_quantile must be between 0 and 1 — skipping")

    # --------------------------------------------------
    # Save mesh
    # --------------------------------------------------
    logger.info("Writing raw mesh to disk")
    o3d.io.write_triangle_mesh(
        str(mesh_out),
        mesh,
        write_ascii=False
    )

    logger.info("Mesh generation completed successfully")
    logger.info(f"Raw mesh saved to: {mesh_out}")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="MARK-2 Mesh Generation")
    parser.add_argument("project_root", type=Path)
    parser.add_argument("--force", action="store_true", help="Overwrite existing mesh output")
    args = parser.parse_args()

    run_mesh_generation(args.project_root, args.force)


if __name__ == "__main__":
    main()
