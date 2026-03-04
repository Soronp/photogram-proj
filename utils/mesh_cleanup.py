#!/usr/bin/env python3
"""
MARK-2 Conditional Mesh Cleanup (Dice to Large Scenes)

Features:
- Inspects mesh_raw before applying any cleanup
- Only applies aggressive operations if problems are detected
- Dynamically adapts to small objects (dice) and large scenes (mosques)
- Corrects inside-out normals only if needed
- Minimal density trimming / planar snapping applied selectively
- Output: mesh_cleaned.ply
"""

from pathlib import Path
import numpy as np
import open3d as o3d
from utils.paths import ProjectPaths
from config_manager import load_config


# --------------------------------------------------
# Lighting-safe normal orientation (conditional)
# --------------------------------------------------
def fix_normals_if_needed(mesh, logger, threshold_ratio=0.1):
    normals = np.asarray(mesh.vertex_normals)
    points = np.asarray(mesh.vertices)
    center = points.mean(axis=0)
    vectors = points - center
    dot = np.einsum("ij,ij->i", normals, vectors)
    inward_ratio = np.sum(dot < 0) / len(dot)
    logger.info(f"[mesh_cleanup] Inward normals ratio: {inward_ratio:.3f}")
    if inward_ratio > threshold_ratio:
        logger.info("[mesh_cleanup] Flipping normals globally")
        normals *= -1
        mesh.vertex_normals = o3d.utility.Vector3dVector(normals)
        mesh.compute_vertex_normals()
    return mesh


# --------------------------------------------------
# Density trimming (conditional)
# --------------------------------------------------
def trim_density_if_needed(mesh, densities, logger, small_object=False, threshold_percent=None):
    densities = np.asarray(densities)
    if threshold_percent is None:
        threshold_percent = 1 if small_object else 5
    # Apply trimming only if there are low-density vertices
    low_density_count = np.sum(densities < np.percentile(densities, threshold_percent))
    if low_density_count / len(densities) > 0.01:  # only trim if >1% vertices are low density
        logger.info(f"[mesh_cleanup] Density trimming: {threshold_percent}%")
        threshold = np.percentile(densities, threshold_percent)
        mask = densities < threshold
        mesh.remove_vertices_by_mask(mask)
        mesh.remove_unreferenced_vertices()
    else:
        logger.info("[mesh_cleanup] Density trimming skipped (mesh looks clean)")
    return mesh


# --------------------------------------------------
# Largest component (conditional)
# --------------------------------------------------
def largest_component_if_needed(mesh, logger, min_components=2):
    clusters, counts, _ = mesh.cluster_connected_triangles()
    num_components = len(counts)
    if num_components >= min_components:
        logger.info(f"[mesh_cleanup] Mesh has {num_components} components, keeping largest")
        largest = np.argmax(counts)
        mesh.remove_triangles_by_mask(clusters != largest)
        mesh.remove_unreferenced_vertices()
    else:
        logger.info(f"[mesh_cleanup] Mesh has {num_components} component(s), skipping largest component filtering")
    return mesh


# --------------------------------------------------
# Optional planar snapping (conditional for small objects)
# --------------------------------------------------
def planar_snap_if_needed(mesh, diag, logger, small_object, tolerance_ratio=0.001):
    if small_object:
        # Check flatness along axes
        points = np.asarray(mesh.vertices)
        flat_axes = [np.std(points[:, axis]) < diag * 0.05 for axis in range(3)]
        if any(flat_axes):
            logger.info("[mesh_cleanup] Applying planar snapping for flat faces")
            for axis in range(3):
                if flat_axes[axis]:
                    unique_vals = np.unique(np.round(points[:, axis] / (diag * tolerance_ratio)) * (diag * tolerance_ratio))
                    for val in unique_vals:
                        mask = np.isclose(points[:, axis], val, atol=diag * tolerance_ratio)
                        points[mask, axis] = val
            mesh.vertices = o3d.utility.Vector3dVector(points)
        else:
            logger.info("[mesh_cleanup] Planar snapping skipped (faces are not flat)")
    return mesh


# --------------------------------------------------
# Run conditional cleanup
# --------------------------------------------------
def run(run_root: Path, project_root: Path, force: bool, logger):
    run_root = Path(run_root).resolve()
    project_root = Path(project_root).resolve()
    paths = ProjectPaths(project_root)
    config = load_config(run_root, logger)
    mesh_cfg = config.get("mesh", {})

    mesh_dir = paths.mesh
    mesh_dir.mkdir(parents=True, exist_ok=True)

    # Load mesh_raw
    mesh_raw_path = mesh_dir / "mesh_raw.ply"
    if not mesh_raw_path.exists():
        raise FileNotFoundError("mesh_raw.ply not found. Generate it before cleanup.")

    mesh = o3d.io.read_triangle_mesh(str(mesh_raw_path))
    if not mesh.has_vertices():
        raise RuntimeError("mesh_raw.ply is empty")

    logger.info(f"[mesh_cleanup] Loaded mesh_raw: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} faces")

    # Determine scale and object type
    bbox = mesh.get_axis_aligned_bounding_box()
    diag = np.linalg.norm(bbox.get_extent())
    small_object = diag < 0.2 or mesh_cfg.get("num_images", 100) < 100
    logger.info(f"[mesh_cleanup] Small object: {small_object}")

    # ----------------------------
    # Normals
    # ----------------------------
    mesh.compute_vertex_normals()  # ensure normals exist
    mesh = fix_normals_if_needed(mesh, logger, threshold_ratio=0.1)

    # ----------------------------
    # Density trimming (Poisson densities required)
    # ----------------------------
    densities_path = mesh_dir / "mesh_raw_densities.npy"
    if densities_path.exists():
        densities = np.load(str(densities_path))
        mesh = trim_density_if_needed(mesh, densities, logger, small_object=small_object)
    else:
        logger.info("[mesh_cleanup] Density trimming skipped (no density info)")

    # ----------------------------
    # Largest component
    # ----------------------------
    mesh = largest_component_if_needed(mesh, logger, min_components=2)

    # ----------------------------
    # Geometry cleanup
    # ----------------------------
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()

    # ----------------------------
    # Planar snapping
    # ----------------------------
    mesh = planar_snap_if_needed(mesh, diag, logger, small_object=small_object)

    # ----------------------------
    # Recompute normals and save
    # ----------------------------
    mesh.compute_vertex_normals()
    output = mesh_dir / "mesh_cleaned.ply"
    o3d.io.write_triangle_mesh(str(output), mesh, write_ascii=False)
    logger.info(f"[mesh_cleanup] Saved mesh_cleaned: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} faces")
    logger.info("[mesh_cleanup] DONE")