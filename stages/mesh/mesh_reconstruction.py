from pathlib import Path
import open3d as o3d
import numpy as np


# =====================================================
# DETERMINISM
# =====================================================
def _set_determinism():
    np.random.seed(42)


# =====================================================
# SCALE ESTIMATION
# =====================================================
def _compute_scale(pcd):
    pts = np.asarray(pcd.points)
    center = pts.mean(axis=0)
    dists = np.linalg.norm(pts - center, axis=1)
    return np.percentile(dists, 90)


# =====================================================
# NORMALS (STABLE + GLOBAL)
# =====================================================
def _compute_normals(pcd, logger):
    logger.info("mesh: computing normals")

    scale = _compute_scale(pcd)
    radius = max(scale * 0.02, 0.01)

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius,
            max_nn=60
        )
    )

    pcd.orient_normals_consistent_tangent_plane(100)

    # Global orientation fix
    pts = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    center = pts.mean(axis=0)

    direction = pts - center
    dot = np.einsum('ij,ij->i', direction, normals)

    if np.mean(dot > 0) < 0.5:
        logger.info("mesh: flipping normals")
        pcd.normals = o3d.utility.Vector3dVector(-normals)

    return pcd


# =====================================================
# POISSON (COMPLETENESS-FIRST)
# =====================================================
def _run_poisson(pcd, logger):
    pts = len(pcd.points)

    # Slightly higher depth for completeness
    if pts < 300_000:
        depth = 10
    elif pts < 1_000_000:
        depth = 11
    elif pts < 3_000_000:
        depth = 12
    else:
        depth = 13

    logger.info(f"mesh: poisson depth = {depth}")

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=depth,
        scale=1.1,         # 🔥 EXPAND surface → fills gaps
        linear_fit=False   # 🔥 more robust than True
    )

    # ❌ NO density filtering → prevents holes

    return mesh


# =====================================================
# CLEAN (SAFE ONLY)
# =====================================================
def _clean(mesh, logger):
    logger.info("mesh: cleaning")

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    return mesh


# =====================================================
# OPTIONAL: LIGHT HOLE FILLING
# =====================================================
def _fill_holes(mesh, logger):
    logger.info("mesh: light smoothing (hole reduction)")

    # VERY light smoothing → helps fill micro gaps
    mesh = mesh.filter_smooth_laplacian(
        number_of_iterations=2
    )

    return mesh


# =====================================================
# MAIN
# =====================================================
def run(paths, config, logger):
    _set_determinism()

    logger.info("---- MESH_RECONSTRUCTION (COMPLETENESS MODE) ----")

    # -------------------------------------------------
    # LOAD (USE BEST FUSED FILE)
    # -------------------------------------------------
    if not paths.fused.exists():
        raise RuntimeError("fused.ply missing")

    pcd = o3d.io.read_point_cloud(str(paths.fused))

    if len(pcd.points) < 5000:
        raise RuntimeError("insufficient points")

    logger.info(f"mesh: input points = {len(pcd.points)}")

    # ❌ NO PRE-CLEAN → preserve full geometry

    # -------------------------------------------------
    # NORMALS
    # -------------------------------------------------
    pcd = _compute_normals(pcd, logger)

    # -------------------------------------------------
    # POISSON
    # -------------------------------------------------
    mesh = _run_poisson(pcd, logger)

    # -------------------------------------------------
    # CLEAN
    # -------------------------------------------------
    mesh = _clean(mesh, logger)

    # -------------------------------------------------
    # LIGHT HOLE FIX
    # -------------------------------------------------
    mesh = _fill_holes(mesh, logger)

    mesh.compute_vertex_normals()

    # -------------------------------------------------
    # SAVE
    # -------------------------------------------------
    o3d.io.write_triangle_mesh(str(paths.mesh_file), mesh)

    logger.info(f"mesh: SUCCESS → {paths.mesh_file}")