#!/usr/bin/env python3
"""
mesh_cleanup.py

MARK-2 Mesh Cleanup Stage (Canonical)
-------------------------------------

- Uses ProjectPaths (single filesystem authority)
- Deterministic
- Resume-safe
- Robust cluster filtering
- Artifact removal via geometric distance + voxel downsampling
"""

from pathlib import Path
import json
import numpy as np
import open3d as o3d

from utils.paths import ProjectPaths
from config_manager import load_config


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def triangle_area(a, b, c) -> float:
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a))


def bbox_volume(verts: np.ndarray) -> float:
    if len(verts) == 0:
        return 0.0
    min_v = verts.min(axis=0)
    max_v = verts.max(axis=0)
    return float(np.prod(max_v - min_v))


def centroid(verts: np.ndarray) -> np.ndarray:
    return verts.mean(axis=0) if len(verts) else np.zeros(3)


def largest_component_mask(mesh: o3d.geometry.TriangleMesh, voxel_size=0.005):
    """
    Downsample mesh, compute connected components, and return a mask
    for triangles belonging to the largest component.
    Memory-safe for large meshes using KD-Tree.
    """
    mesh_vertices = np.asarray(mesh.vertices)

    # Sample points for DBSCAN
    pcd = mesh.sample_points_uniformly(number_of_points=len(mesh_vertices))
    pcd = pcd.voxel_down_sample(voxel_size)
    pcd_vertices = np.asarray(pcd.points)

    labels = np.array(pcd.cluster_dbscan(eps=voxel_size * 2, min_points=10))
    if len(labels) == 0 or np.max(labels) < 0:
        return np.ones(len(mesh.triangles), dtype=bool)

    largest_label = np.bincount(labels[labels >= 0]).argmax()

    # Build KD-Tree for pcd points
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    # Mask vertices belonging to largest cluster
    mask_vertices = np.zeros(len(mesh_vertices), dtype=bool)
    for i, v in enumerate(mesh_vertices):
        [_, idx, _] = kdtree.search_knn_vector_3d(v, 1)
        if labels[idx[0]] == largest_label:
            mask_vertices[i] = True

    # Mask triangles whose all vertices belong to largest component
    triangles = np.asarray(mesh.triangles)
    mask_triangles = mask_vertices[triangles].all(axis=1)
    return mask_triangles
# ------------------------------------------------------------
# Stage
# ------------------------------------------------------------

def run(run_root: Path, project_root: Path, force: bool, logger):

    run_root = Path(run_root).resolve()
    project_root = Path(project_root).resolve()

    paths = ProjectPaths(project_root)
    config = load_config(run_root, logger)
    mesh_cfg = config.get("mesh", {})

    logger.info("[mesh_cleanup] START")

    mesh_dir = paths.mesh
    mesh_in = mesh_dir / "mesh_raw.ply"
    mesh_out = mesh_dir / "mesh_cleaned.ply"
    report_out = mesh_dir / "mesh_cleanup_report.json"

    if not mesh_in.exists():
        raise FileNotFoundError(f"[mesh_cleanup] Missing mesh: {mesh_in}")

    if mesh_out.exists() and not force:
        logger.info("[mesh_cleanup] Output exists — skipping")
        return

    mesh = o3d.io.read_triangle_mesh(str(mesh_in))
    if not mesh.has_triangles():
        raise RuntimeError("[mesh_cleanup] Mesh contains no triangles")

    mesh.compute_vertex_normals()

    # ------------------------------------------------------------
    # Basic topology cleanup
    # ------------------------------------------------------------

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()

    logger.info(f"[mesh_cleanup] Triangles after topo cleanup: {len(mesh.triangles):,}")

    # ------------------------------------------------------------
    # Artifact removal via largest component
    # ------------------------------------------------------------

    voxel_size = float(mesh_cfg.get("artifact_voxel_size", 0.002))
    mask_triangles = largest_component_mask(mesh, voxel_size=voxel_size)
    if mask_triangles.sum() < len(mesh.triangles):
        logger.info(f"[mesh_cleanup] Removing {len(mesh.triangles) - mask_triangles.sum():,} artifact triangles")
        mesh.remove_triangles_by_index(np.where(~mask_triangles)[0])
        mesh.remove_unreferenced_vertices()

    # ------------------------------------------------------------
    # Optional smoothing
    # ------------------------------------------------------------

    if mesh_cfg.get("smoothing", False):
        iters = int(mesh_cfg.get("smoothing_iterations", 5))
        mesh = mesh.filter_smooth_laplacian(iters)

    # ------------------------------------------------------------
    # Optional decimation
    # ------------------------------------------------------------

    ratio = mesh_cfg.get("decimation_ratio")
    if ratio is not None and 0.0 < float(ratio) < 1.0:
        target = int(len(mesh.triangles) * float(ratio))
        mesh = mesh.simplify_quadric_decimation(target)

    mesh.compute_vertex_normals()

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------

    o3d.io.write_triangle_mesh(str(mesh_out), mesh, write_ascii=False)

    with open(report_out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "input_triangles": len(np.asarray(mesh.triangles)),
                "output_triangles": len(np.asarray(mesh.triangles)),
                "deterministic": True,
            },
            f,
            indent=2,
        )

    logger.info("[mesh_cleanup] DONE")