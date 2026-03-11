"""
mesh_cleanup.py

Stage — Mesh cleanup

Performs post-processing on reconstructed meshes to remove
artifacts commonly produced by MVS reconstruction.

Operations
----------
• conditional normal correction
• density trimming
• small component removal
• thin triangle removal (edge spikes)
• planar snapping
• optional smoothing
"""

from pathlib import Path
import numpy as np
import open3d as o3d


# --------------------------------------------------
# Normal correction
# --------------------------------------------------

def fix_normals(mesh, logger, threshold=0.1):

    mesh.compute_vertex_normals()

    normals = np.asarray(mesh.vertex_normals)
    points = np.asarray(mesh.vertices)

    center = points.mean(axis=0)

    dot = np.einsum("ij,ij->i", normals, points - center)

    inward_ratio = np.sum(dot < 0) / len(dot)

    logger.info(f"[mesh_cleanup] inward normals ratio: {inward_ratio:.3f}")

    if inward_ratio > threshold:
        normals *= -1
        mesh.vertex_normals = o3d.utility.Vector3dVector(normals)
        mesh.compute_vertex_normals()
        logger.info("[mesh_cleanup] normals flipped")

    return mesh


# --------------------------------------------------
# Density trimming
# --------------------------------------------------

def trim_density(mesh, densities, logger, small_object):

    densities = np.asarray(densities)

    percentile = 1 if small_object else 5

    cutoff = np.percentile(densities, percentile)

    mask = densities < cutoff

    removed = mask.sum()

    if removed > 0:
        mesh.remove_vertices_by_mask(mask)
        mesh.remove_unreferenced_vertices()
        logger.info(f"[mesh_cleanup] removed {removed} low-density vertices")

    return mesh


# --------------------------------------------------
# Small cluster removal
# --------------------------------------------------

def remove_small_clusters(mesh, logger, min_triangles):

    clusters, counts, _ = mesh.cluster_connected_triangles()

    counts = np.asarray(counts)

    keep = counts >= min_triangles

    if not keep.any():
        return mesh

    mask = np.array([keep[c] for c in clusters])

    mesh.remove_triangles_by_mask(~mask)
    mesh.remove_unreferenced_vertices()

    logger.info(
        f"[mesh_cleanup] removed {len(counts) - keep.sum()} small clusters"
    )

    return mesh


# --------------------------------------------------
# Thin triangle removal
# --------------------------------------------------

def remove_thin_triangles(mesh, logger, aspect_threshold):

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    remove = []

    for i, tri in enumerate(triangles):

        pts = vertices[tri]

        e = [
            np.linalg.norm(pts[0] - pts[1]),
            np.linalg.norm(pts[1] - pts[2]),
            np.linalg.norm(pts[2] - pts[0]),
        ]

        s = sum(e) / 2

        area = max(
            np.sqrt(s * (s - e[0]) * (s - e[1]) * (s - e[2])),
            1e-12
        )

        h_min = 2 * area / max(e)

        aspect = max(e) / h_min

        if aspect > aspect_threshold:
            remove.append(i)

    if remove:

        mask = np.ones(len(triangles), dtype=bool)
        mask[remove] = False

        mesh.remove_triangles_by_mask(~mask)
        mesh.remove_unreferenced_vertices()

        logger.info(f"[mesh_cleanup] removed {len(remove)} thin triangles")

    return mesh


# --------------------------------------------------
# Largest component
# --------------------------------------------------

def keep_largest_component(mesh, logger):

    clusters, counts, _ = mesh.cluster_connected_triangles()

    counts = np.asarray(counts)

    if len(counts) <= 1:
        return mesh

    largest = np.argmax(counts)

    mask = np.array([c == largest for c in clusters])

    mesh.remove_triangles_by_mask(~mask)
    mesh.remove_unreferenced_vertices()

    logger.info(f"[mesh_cleanup] kept largest component")

    return mesh


# --------------------------------------------------
# Planar snapping
# --------------------------------------------------

def planar_snap(mesh, diag, logger, small_object):

    if not small_object:
        return mesh

    pts = np.asarray(mesh.vertices)

    flat_axes = [np.std(pts[:, i]) < diag * 0.05 for i in range(3)]

    if not any(flat_axes):
        return mesh

    logger.info("[mesh_cleanup] planar snapping")

    tol = diag * 0.001

    for axis in range(3):

        if not flat_axes[axis]:
            continue

        vals = np.round(pts[:, axis] / tol) * tol

        pts[:, axis] = vals

    mesh.vertices = o3d.utility.Vector3dVector(pts)

    return mesh


# --------------------------------------------------
# Smoothing
# --------------------------------------------------

def smooth(mesh, logger, iterations):

    if iterations <= 0:
        return mesh

    mesh = mesh.filter_smooth_simple(number_of_iterations=iterations)

    logger.info(f"[mesh_cleanup] smoothing iterations: {iterations}")

    return mesh


# --------------------------------------------------
# Stage execution
# --------------------------------------------------

def run(paths, logger, tools, config):

    logger.info("[mesh_cleanup] starting")

    mesh_dir = paths.mesh

    raw_mesh = mesh_dir / "mesh_raw.ply"

    if not raw_mesh.exists():
        raise FileNotFoundError("mesh_raw.ply not found")

    mesh = o3d.io.read_triangle_mesh(str(raw_mesh))

    if not mesh.has_vertices():
        raise RuntimeError("mesh is empty")

    logger.info(
        f"[mesh_cleanup] mesh loaded: "
        f"{len(mesh.vertices):,} vertices / {len(mesh.triangles):,} faces"
    )

    bbox = mesh.get_axis_aligned_bounding_box()

    diag = np.linalg.norm(bbox.get_extent())

    small_object = diag < 0.2

    logger.info(f"[mesh_cleanup] small object: {small_object}")

    mesh = fix_normals(mesh, logger)

    density_file = mesh_dir / "mesh_raw_densities.npy"

    if density_file.exists():
        densities = np.load(density_file)
        mesh = trim_density(mesh, densities, logger, small_object)

    mesh = remove_small_clusters(
        mesh,
        logger,
        min_triangles=10 if small_object else 50
    )

    mesh = remove_thin_triangles(
        mesh,
        logger,
        aspect_threshold=10 if small_object else 15
    )

    mesh = keep_largest_component(mesh, logger)

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()

    mesh = planar_snap(mesh, diag, logger, small_object)

    mesh = smooth(mesh, logger, iterations=1 if small_object else 0)

    mesh.compute_vertex_normals()

    output = mesh_dir / "mesh_cleaned.ply"

    o3d.io.write_triangle_mesh(str(output), mesh)

    logger.info(f"[mesh_cleanup] saved → {output.name}")

    logger.info("[mesh_cleanup] completed")