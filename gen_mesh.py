#!/usr/bin/env python3
"""
gen_mesh.py (Gentle Mesh Generation)

MARK-2 Mesh Generation Stage (Non-Aggressive)
--------------------------------------------

- Uses cleaned dense cloud if available, else raw fused.ply
- Minimal Poisson reconstruction for patching only
- Preserves original cloud geometry
- Optional light micro-hole filling
- Optional light edge trimming
- Deterministic & resume-safe
"""

from pathlib import Path
import shutil
import numpy as np
import open3d as o3d

from utils.paths import ProjectPaths
from utils.config import load_config

# --------------------------------------------------
# Helpers
# --------------------------------------------------

def remove_edge_artifacts(mesh: o3d.geometry.TriangleMesh, area_ratio=1e-5, distance_ratio=0.25):
    """
    Lightly remove extreme tiny/far triangles along edges.
    Skips if more than 80% of triangles would be removed.
    """
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)
    if len(triangles) == 0:
        return mesh

    bbox = mesh.get_axis_aligned_bounding_box()
    diag = np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound())
    area_threshold = (diag ** 2) * area_ratio
    distance_threshold = diag * distance_ratio

    tris_pts = vertices[triangles]
    vec0 = tris_pts[:, 1] - tris_pts[:, 0]
    vec1 = tris_pts[:, 2] - tris_pts[:, 0]
    tri_area = 0.5 * np.linalg.norm(np.cross(vec0, vec1), axis=1)

    centroid = vertices.mean(axis=0)
    tri_centroids = tris_pts.mean(axis=1)
    dist = np.linalg.norm(tri_centroids - centroid, axis=1)

    keep_mask = (tri_area > area_threshold) & (dist < distance_threshold)
    kept = np.count_nonzero(keep_mask)

    if kept < len(triangles) * 0.2:
        return mesh

    mesh.remove_triangles_by_mask(~keep_mask)
    mesh.remove_unreferenced_vertices()
    return mesh

def fill_micro_holes(mesh: o3d.geometry.TriangleMesh, max_hole_edges=4):
    """
    Fill only very small holes (<= max_hole_edges)
    """
    mesh.compute_vertex_normals()
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    if len(triangles) == 0:
        return mesh

    edges = mesh.get_non_manifold_edges(allow_boundary_edges=True)
    if len(edges) == 0:
        return mesh

    edge_map = {}
    for e in edges:
        edge_map.setdefault(e[0], []).append(e[1])
        edge_map.setdefault(e[1], []).append(e[0])

    visited = set()
    new_vertices = vertices.tolist()
    new_triangles = triangles.tolist()

    for start in edge_map:
        if start in visited:
            continue
        loop = [start]
        current, prev = start, None
        while True:
            neighbors = [n for n in edge_map[current] if n != prev]
            if not neighbors:
                break
            next_v = neighbors[0]
            if next_v in loop:
                break
            loop.append(next_v)
            prev, current = current, next_v
        visited.update(loop)

        if 3 <= len(loop) <= max_hole_edges:
            center = np.mean([new_vertices[i] for i in loop], axis=0)
            center_idx = len(new_vertices)
            new_vertices.append(center.tolist())
            for i in range(len(loop)):
                new_triangles.append([loop[i], loop[(i + 1) % len(loop)], center_idx])

    mesh.vertices = o3d.utility.Vector3dVector(np.array(new_vertices))
    mesh.triangles = o3d.utility.Vector3iVector(np.array(new_triangles))
    mesh.remove_duplicated_triangles()
    mesh.remove_unreferenced_vertices()
    return mesh

# --------------------------------------------------
# Main Stage
# --------------------------------------------------

def run(run_root: Path, project_root: Path, force: bool, logger):
    run_root = Path(run_root).resolve()
    project_root = Path(project_root).resolve()

    paths = ProjectPaths(project_root)
    config = load_config(run_root)
    mesh_cfg = config.get("mesh", {})

    logger.info("[mesh] START")

    dense_dir = paths.dense
    mesh_dir = paths.mesh

    raw_ply = dense_dir / "fused.ply"
    cleaned_ply = dense_dir / "fused_cleaned.ply"
    mesh_out = mesh_dir / "mesh_raw.ply"

    # Select input cloud
    if cleaned_ply.exists():
        input_cloud = cleaned_ply
        logger.info("[mesh] Using cleaned dense cloud")
    elif raw_ply.exists():
        input_cloud = raw_ply
        logger.warning("[mesh] Cleanup missing — using raw fused.ply")
    else:
        raise FileNotFoundError(f"[mesh] Dense cloud not found:\n  {cleaned_ply}\n  {raw_ply}")

    logger.info(f"[mesh] Input : {input_cloud}")
    logger.info(f"[mesh] Output: {mesh_out}")

    # Skip logic
    if mesh_out.exists() and not force:
        logger.info("[mesh] Output exists — skipping")
        return
    if mesh_dir.exists() and force:
        shutil.rmtree(mesh_dir)
    mesh_dir.mkdir(parents=True, exist_ok=True)

    # Load point cloud
    pcd = o3d.io.read_point_cloud(str(input_cloud))
    if not pcd.has_points():
        raise RuntimeError("[mesh] Input point cloud is empty")
    pcd.remove_non_finite_points()
    logger.info(f"[mesh] Points: {len(pcd.points):,}")

    # Estimate normals
    bbox = pcd.get_axis_aligned_bounding_box()
    diag = np.linalg.norm(bbox.get_extent())
    if diag <= 0:
        raise RuntimeError("[mesh] Degenerate geometry (invalid bounding box)")

    radius = diag * 0.02  # slightly larger radius
    max_nn = 20           # smaller neighborhood for gentler normals
    logger.info(f"[mesh] Estimating normals (radius={radius:.6f}, max_nn={max_nn})")
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    pcd.orient_normals_consistent_tangent_plane(k=20)
    pcd.normalize_normals()

    # Optional: lightly clean outliers
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=15, std_ratio=2.0)
    pcd, _ = pcd.remove_radius_outlier(nb_points=3, radius=diag*0.01)
    logger.info(f"[mesh] Cleaned points: {len(pcd.points):,}")

    # Poisson reconstruction (shallow, gentle)
    depth = int(mesh_cfg.get("poisson_depth", 8))  # lower depth to avoid patchy reconstruction
    logger.info(f"[mesh] Poisson reconstruction (depth={depth})")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    if not mesh.has_triangles():
        raise RuntimeError("[mesh] Poisson reconstruction failed")
    logger.info(f"[mesh] Raw mesh: {len(mesh.vertices):,} vertices | {len(mesh.triangles):,} triangles")

    # Optional: very light micro-hole filling
    mesh = fill_micro_holes(mesh, max_hole_edges=4)

    # Optional: very light edge artifact trimming
    mesh = remove_edge_artifacts(mesh, area_ratio=5e-6, distance_ratio=0.3)

    # Final cleanup
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()

    logger.info(f"[mesh] Final mesh: {len(mesh.vertices):,} vertices | {len(mesh.triangles):,} triangles")

    # Save
    o3d.io.write_triangle_mesh(str(mesh_out), mesh, write_ascii=False)
    logger.info(f"[mesh] Mesh written: {mesh_out}")
    logger.info("[mesh] DONE")