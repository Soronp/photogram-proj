#!/usr/bin/env python3
"""
mesh_cleanup.py

MARK-2 Mesh Cleanup Stage (Dual-Poisson + Robust Artifact & Micro-hole Removal)
-------------------------------------------------------------------------------

- Removes Poisson halo / sandy edge artifacts including curved boundaries
- Fills micro holes deterministically (small loops + planar projection)
- Preserves real surface geometry
- Generates metrics compatible with mesh evaluation
- Deterministic & resume-safe
"""

from pathlib import Path
import json
import numpy as np
import open3d as o3d
from scipy.spatial import Delaunay

from utils.paths import ProjectPaths
from config_manager import load_config


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def resolve_fused_cloud(paths: ProjectPaths, logger) -> Path:
    candidates = [paths.dense / "fused_cleaned.ply", paths.dense / "fused.ply"]
    for p in candidates:
        if p.exists():
            logger.info(f"[mesh_cleanup] Using fused cloud: {p}")
            return p
    raise FileNotFoundError(
        "[mesh_cleanup] Missing fused point cloud.\n"
        f"Expected one of:\n  {candidates[0]}\n  {candidates[1]}"
    )

def triangle_area(tri_pts: np.ndarray) -> np.ndarray:
    """Compute triangle areas for an array of triangle vertices."""
    vec0 = tri_pts[:,1] - tri_pts[:,0]
    vec1 = tri_pts[:,2] - tri_pts[:,0]
    return 0.5 * np.linalg.norm(np.cross(vec0, vec1), axis=1)

def remove_degenerate(mesh: o3d.geometry.TriangleMesh, eps=1e-12):
    """Remove triangles with near-zero area."""
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)
    if len(triangles) == 0:
        return mesh
    tri_pts = vertices[triangles]
    areas = triangle_area(tri_pts)
    mask = areas > eps
    mesh.remove_triangles_by_mask(~mask)
    mesh.remove_unreferenced_vertices()
    return mesh

def fill_micro_holes(mesh: o3d.geometry.TriangleMesh, max_hole_edges=12):
    """Fill small boundary loops deterministically using planar projection."""
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
        # Traverse boundary loop
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
            # Planar projection fill
            pts = np.array([new_vertices[i] for i in loop])
            center = pts.mean(axis=0)
            new_vertices.append(center.tolist())
            center_idx = len(new_vertices) - 1
            for i in range(len(loop)):
                new_triangles.append([loop[i], loop[(i+1)%len(loop)], center_idx])
        elif len(loop) > max_hole_edges:
            # Planar Delaunay for larger loops
            pts = np.array([new_vertices[i] for i in loop])
            pts_mean = pts.mean(axis=0)
            cov = np.cov((pts - pts_mean).T)
            eigvals, eigvecs = np.linalg.eigh(cov)
            u, v = eigvecs[:,1], eigvecs[:,2]
            proj = np.dot(pts - pts_mean, np.column_stack((u,v)))
            tri = Delaunay(proj)
            for simplex in tri.simplices:
                new_triangles.append([loop[simplex[0]], loop[simplex[1]], loop[simplex[2]]])

    mesh.vertices = o3d.utility.Vector3dVector(np.array(new_vertices))
    mesh.triangles = o3d.utility.Vector3iVector(np.array(new_triangles))
    mesh.remove_duplicated_triangles()
    mesh.remove_unreferenced_vertices()
    return mesh

def remove_edge_artifacts(mesh: o3d.geometry.TriangleMesh, area_ratio=1e-6, distance_ratio=0.15, curvature_ratio=0.85, iterations=2):
    """Iteratively remove halo/edge triangles based on area, distance, and curvature."""
    for _ in range(iterations):
        triangles = np.asarray(mesh.triangles)
        vertices = np.asarray(mesh.vertices)
        if len(triangles) == 0:
            break

        bbox = mesh.get_axis_aligned_bounding_box()
        diag = np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound())
        area_threshold = (diag**2)*area_ratio
        distance_threshold = diag*distance_ratio

        tris_pts = vertices[triangles]
        tri_area = 0.5*np.linalg.norm(np.cross(tris_pts[:,1]-tris_pts[:,0], tris_pts[:,2]-tris_pts[:,0]), axis=1)

        centroid = vertices.mean(axis=0)
        tri_centroids = tris_pts.mean(axis=1)
        dist = np.linalg.norm(tri_centroids - centroid, axis=1)

        mesh.compute_vertex_normals()
        normals = np.asarray(mesh.vertex_normals)
        tri_normals = normals[triangles].mean(axis=1)
        normal_dev = np.linalg.norm(tri_normals, axis=1)

        keep_mask = (tri_area > area_threshold) & (dist < distance_threshold) & (normal_dev < curvature_ratio)
        if np.count_nonzero(keep_mask) < len(triangles)*0.1:
            break
        mesh.remove_triangles_by_mask(~keep_mask)
        mesh.remove_unreferenced_vertices()
    return mesh

def full_topology_cleanup(mesh: o3d.geometry.TriangleMesh):
    mesh = remove_degenerate(mesh)
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    return mesh

def largest_component(mesh: o3d.geometry.TriangleMesh):
    clusters, cluster_counts, _ = mesh.cluster_connected_triangles()
    if len(cluster_counts) == 0:
        return mesh
    largest_idx = np.argmax(cluster_counts)
    remove_mask = clusters != largest_idx
    mesh.remove_triangles_by_mask(remove_mask)
    mesh.remove_unreferenced_vertices()
    return mesh

def boundary_aware_smoothing(mesh: o3d.geometry.TriangleMesh, taubin_iters=2, laplacian_iters=2):
    mesh.filter_smooth_taubin(number_of_iterations=taubin_iters)
    if len(mesh.get_non_manifold_edges(allow_boundary_edges=True)) > 0:
        mesh.filter_smooth_laplacian(number_of_iterations=laplacian_iters)
    return mesh


# ------------------------------------------------------------
# Main Stage
# ------------------------------------------------------------
def run(run_root: Path, project_root: Path, force: bool, logger):
    run_root = Path(run_root).resolve()
    project_root = Path(project_root).resolve()

    paths = ProjectPaths(project_root)
    config = load_config(run_root, logger)
    mesh_cfg = config.get("mesh", {})

    mesh_dir = paths.mesh
    mesh_out = mesh_dir / "mesh_cleaned.ply"
    report_out = mesh_dir / "mesh_cleanup_report.json"

    if mesh_out.exists() and not force:
        logger.info("[mesh_cleanup] Output exists — skipping")
        return

    mesh_dir.mkdir(parents=True, exist_ok=True)

    # --- Load fused cloud ---
    fused_path = resolve_fused_cloud(paths, logger)
    pcd = o3d.io.read_point_cloud(str(fused_path))
    if not pcd.has_points():
        raise RuntimeError("[mesh_cleanup] Fused cloud contains no points")

    depth_high = int(mesh_cfg.get("poisson_depth", 10))
    depth_low = int(mesh_cfg.get("poisson_low_depth", max(depth_high-2,6)))
    trim_q = float(mesh_cfg.get("density_trim_quantile", 0.04))
    smooth_iters = int(mesh_cfg.get("smoothing_iterations", 2))
    decimation_ratio = mesh_cfg.get("decimation_ratio")

    # --- Dual Poisson reconstruction ---
    logger.info("[mesh_cleanup] Low-depth Poisson (edges)")
    mesh_low, densities_low = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth_low, linear_fit=True)
    mesh_low.compute_vertex_normals()

    logger.info("[mesh_cleanup] High-depth Poisson (interior)")
    mesh_high, densities_high = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth_high, linear_fit=True)
    mesh_high.compute_vertex_normals()

    # --- Merge ---
    mesh = mesh_low + mesh_high
    mesh = full_topology_cleanup(mesh)
    logger.info(f"[mesh_cleanup] After merge: {len(mesh.triangles):,} triangles")

    # --- Edge/Halo removal ---
    mesh = remove_edge_artifacts(mesh)
    logger.info(f"[mesh_cleanup] After edge removal: {len(mesh.triangles):,} triangles")

    # --- Fill micro holes ---
    mesh = fill_micro_holes(mesh)
    logger.info(f"[mesh_cleanup] After hole filling: {len(mesh.triangles):,} triangles")

    # --- Largest component ---
    mesh = largest_component(mesh)
    logger.info(f"[mesh_cleanup] After largest component: {len(mesh.triangles):,} triangles")

    # --- Boundary-aware smoothing ---
    mesh = boundary_aware_smoothing(mesh, taubin_iters=smooth_iters)
    mesh = full_topology_cleanup(mesh)
    logger.info(f"[mesh_cleanup] Applied smoothing ({smooth_iters} iterations)")

    # --- Optional decimation ---
    if decimation_ratio is not None:
        ratio = float(decimation_ratio)
        if 0.0 < ratio < 1.0:
            target = int(len(mesh.triangles) * ratio)
            mesh = mesh.simplify_quadric_decimation(target)
            mesh = full_topology_cleanup(mesh)
            logger.info(f"[mesh_cleanup] Decimated to {target:,} triangles")

    mesh.compute_vertex_normals()

    # --- Save output ---
    if len(mesh.triangles) == 0 or len(mesh.vertices) == 0:
        logger.warning("[mesh_cleanup] Mesh empty — skipping write")
    else:
        o3d.io.write_triangle_mesh(str(mesh_out), mesh, write_ascii=False)
        logger.info(f"[mesh_cleanup] Mesh written: {mesh_out}")

    # --- Save metrics for evaluation ---
    metrics = {
        "output_triangles": len(mesh.triangles),
        "output_vertices": len(mesh.vertices),
        "poisson_high_depth": depth_high,
        "poisson_low_depth": depth_low,
        "density_trim_quantile": trim_q,
        "dual_poisson_merge": True,
        "edge_artifact_removal": True,
        "micro_hole_filling": True,
        "largest_component": True,
        "boundary_smoothing": True,
        "deterministic": True
    }

    with open(report_out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"[mesh_cleanup] Cleanup report written: {report_out}")
    logger.info("[mesh_cleanup] DONE")