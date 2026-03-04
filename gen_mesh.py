#!/usr/bin/env python3
"""
gen_mesh_dice_smart.py (MARK-2 Dynamic Dice Surface Reconstruction)

Features:
- Automatically chooses Poisson depth, trimming, and planar handling
- Preserves cube sides without destructive plane attenuation
- Minimal, adaptive density trimming
- Lighting-safe normals
- Output saved as mesh_raw.ply (ready for optional mesh cleanup)
- Deterministic and concise
"""

from pathlib import Path
import shutil
import numpy as np
import open3d as o3d

from utils.paths import ProjectPaths
from utils.config import load_config


# --------------------------------------------------
# Lighting-Safe Normal Orientation
# --------------------------------------------------
def orient_normals_smart(pcd, logger):
    pcd.orient_normals_consistent_tangent_plane(k=50)
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    center = points.mean(axis=0)
    vectors = points - center
    dot = np.einsum("ij,ij->i", normals, vectors)
    if np.sum(dot < 0) / len(dot) > 0.5:
        normals *= -1
    pcd.normals = o3d.utility.Vector3dVector(normals)
    pcd.normalize_normals()
    return pcd


# --------------------------------------------------
# Density Trimming (adaptive)
# --------------------------------------------------
def adaptive_density_trim(mesh, densities, logger, min_trim=0, max_trim=5):
    densities = np.asarray(densities)
    # Determine trim percent based on mesh size and flatness
    avg_density = np.mean(densities)
    trim_percent = max(min(max_trim, int(0.01 * avg_density)), min_trim)
    logger.info(f"[mesh] Adaptive density trim: {trim_percent}%")
    threshold = np.percentile(densities, trim_percent)
    mask = densities < threshold
    mesh.remove_vertices_by_mask(mask)
    mesh.remove_unreferenced_vertices()
    return mesh


# --------------------------------------------------
# Optional planar snapping for dice
# --------------------------------------------------
def snap_to_planes(mesh, diag, logger, tolerance_ratio=0.001):
    tolerance = diag * tolerance_ratio
    points = np.asarray(mesh.vertices)
    for axis in range(3):
        unique_vals = np.unique(np.round(points[:, axis] / tolerance) * tolerance)
        for val in unique_vals:
            mask = np.isclose(points[:, axis], val, atol=tolerance)
            points[mask, axis] = val
    mesh.vertices = o3d.utility.Vector3dVector(points)
    return mesh


# --------------------------------------------------
# Main dynamic reconstruction
# --------------------------------------------------
def run(run_root: Path, project_root: Path, force: bool, logger):
    run_root = Path(run_root).resolve()
    project_root = Path(project_root).resolve()
    paths = ProjectPaths(project_root)
    config = load_config(run_root)
    mesh_cfg = config.get("mesh", {})

    dense_dir = paths.dense
    mesh_dir = paths.mesh
    mesh_dir.mkdir(parents=True, exist_ok=True)

    raw_ply = dense_dir / "fused.ply"
    cleaned_ply = dense_dir / "fused_cleaned.ply"
    mesh_out = mesh_dir / "mesh_raw.ply"

    if cleaned_ply.exists():
        input_cloud = cleaned_ply
    elif raw_ply.exists():
        input_cloud = raw_ply
    else:
        raise FileNotFoundError("Dense cloud missing")

    if mesh_out.exists() and not force:
        logger.info("[mesh] Output exists — skipping")
        return

    if mesh_dir.exists() and force:
        shutil.rmtree(mesh_dir)
        mesh_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # Load cloud
    # ----------------------------
    pcd = o3d.io.read_point_cloud(str(input_cloud))
    pcd.remove_non_finite_points()
    logger.info(f"[mesh] Points loaded: {len(pcd.points):,}")

    # ----------------------------
    # Normal estimation
    # ----------------------------
    bbox = pcd.get_axis_aligned_bounding_box()
    diag = np.linalg.norm(bbox.get_extent())
    radius = diag * 0.01
    max_nn = 50
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    pcd = orient_normals_smart(pcd, logger)

    # ----------------------------
    # Poisson reconstruction (dynamic depth)
    # ----------------------------
    depth = int(mesh_cfg.get("poisson_depth", 12))
    if diag < 0.1:  # very small objects like dice
        depth = max(depth, 12)
    logger.info(f"[mesh] Poisson reconstruction (depth={depth})")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    logger.info(f"[mesh] Raw mesh: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} faces")

    # ----------------------------
    # Adaptive density trimming
    # ----------------------------
    mesh = adaptive_density_trim(mesh, densities, logger)

    # ----------------------------
    # Largest component
    # ----------------------------
    labels = np.array(mesh.cluster_connected_triangles()[0])
    largest_label = np.bincount(labels).argmax()
    mesh.remove_triangles_by_mask(labels != largest_label)
    mesh.remove_unreferenced_vertices()

    # ----------------------------
    # Cleanup
    # ----------------------------
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()

    # ----------------------------
    # Optional planar snapping (only for very small objects)
    # ----------------------------
    if diag < 0.15:  # threshold for dice
        logger.info("[mesh] Applying planar snapping for cube sides")
        mesh = snap_to_planes(mesh, diag, logger)

    mesh.compute_vertex_normals()
    logger.info(f"[mesh] Final mesh: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} faces")

    o3d.io.write_triangle_mesh(str(mesh_out), mesh, write_ascii=False)
    logger.info("[mesh] DONE")