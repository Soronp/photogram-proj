#!/usr/bin/env python3
"""
gen_mesh.py

MARK-2 Mesh Generation Stage
----------------------------
- Generates a raw surface mesh from dense point cloud
- Algorithm: Poisson (deterministic)
- Input: dense/fused.ply
- Output: mesh/mesh_raw.ply
- Fully logged, restart-safe
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
# Mesh generation
# ------------------------------------------------------------------

def run_mesh_generation(project_root: Path, force: bool):
    paths = ProjectPaths(project_root)
    config = load_config(project_root)
    logger = get_logger("gen_mesh", project_root)

    dense_ply = paths.dense / "fused.ply"
    mesh_dir = paths.mesh
    mesh_out = mesh_dir / "mesh_raw.ply"

    logger.info("=== Mesh Generation Stage ===")
    logger.info(f"Dense input : {dense_ply}")
    logger.info(f"Mesh output : {mesh_out}")

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
        raise RuntimeError("Dense point cloud is empty")

    logger.info(f"Loaded {len(pcd.points):,} points")

    # Remove NaNs / invalid points
    pcd.remove_non_finite_points()
    logger.info(f"After cleanup: {len(pcd.points):,} points")

    # --------------------------------------------------
    # Normal estimation (required for Poisson)
    # --------------------------------------------------

    logger.info("Estimating normals")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=float(config.get("mesh_normal_radius", 0.1)),
            max_nn=int(config.get("mesh_normal_max_nn", 30))
        )
    )
    pcd.normalize_normals()

    # --------------------------------------------------
    # Poisson surface reconstruction
    # --------------------------------------------------

    depth = int(config.get("mesh_poisson_depth", 10))
    logger.info(f"Running Poisson reconstruction (depth={depth})")

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=depth
    )

    if not mesh.has_triangles():
        raise RuntimeError("Poisson reconstruction failed — no triangles produced")

    logger.info(f"Initial mesh: {len(mesh.vertices):,} vertices, "
                f"{len(mesh.triangles):,} triangles")

    # --------------------------------------------------
    # Optional density-based vertex trimming
    # --------------------------------------------------

    density_quantile = config.get("mesh_density_quantile")

    if density_quantile is not None:
        q = float(density_quantile)
        if not (0.0 < q < 1.0):
            logger.warning("mesh_density_quantile must be between 0 and 1 — skipping trim")
        else:
            logger.info(f"Applying density trimming (quantile={q})")

            densities = np.asarray(densities)
            cutoff = np.quantile(densities, q)

            remove_mask = densities < cutoff
            removed = int(np.sum(remove_mask))

            mesh.remove_vertices_by_mask(remove_mask)

            logger.info(f"Removed {removed:,} low-density vertices")
            logger.info(f"Trimmed mesh: {len(mesh.vertices):,} vertices, "
                        f"{len(mesh.triangles):,} triangles")

    # --------------------------------------------------
    # Save mesh
    # --------------------------------------------------

    logger.info("Writing raw mesh to disk")
    o3d.io.write_triangle_mesh(str(mesh_out), mesh, write_ascii=False)

    logger.info("Mesh generation completed successfully")
    logger.info(f"Raw mesh saved to: {mesh_out}")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MARK-2 Mesh Generation")
    parser.add_argument("project_root", type=Path)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing mesh output"
    )

    args = parser.parse_args()
    run_mesh_generation(args.project_root, args.force)


if __name__ == "__main__":
    main()
