from pathlib import Path
import open3d as o3d
import numpy as np


def _enforce_outward_normals(pcd, logger):
    logger.info("mesh: enforcing global outward normals")

    center = pcd.get_axis_aligned_bounding_box().get_center()

    pts = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)

    directions = pts - center
    dot = np.einsum("ij,ij->i", normals, directions)

    flip_mask = dot < 0
    normals[flip_mask] *= -1

    pcd.normals = o3d.utility.Vector3dVector(normals)
    return pcd


def run(paths, config, logger):
    stage = "mesh_reconstruction"
    logger.info(f"---- {stage.upper()} ----")

    input_ply = paths.fused
    output_mesh = paths.mesh_file

    if not input_ply.exists():
        raise RuntimeError(f"{stage}: fused.ply not found")

    if output_mesh.exists():
        logger.warning(f"{stage}: removing previous mesh")
        output_mesh.unlink()

    # =====================================================
    # LOAD
    # =====================================================
    logger.info("mesh: loading point cloud")
    pcd = o3d.io.read_point_cloud(str(input_ply))

    if len(pcd.points) < 5000:
        raise RuntimeError("mesh: insufficient points")

    logger.info(f"mesh: points = {len(pcd.points)}")

    # =====================================================
    # LIGHT DOWNSAMPLE (ONLY IF HUGE)
    # =====================================================
    if len(pcd.points) > 1_500_000:
        voxel = 0.0025
        logger.info(f"mesh: voxel downsample = {voxel}")
        pcd = pcd.voxel_down_sample(voxel)

    # =====================================================
    # EDGE-SAFE DENOISE
    # =====================================================
    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=16,
        std_ratio=3.0
    )

    logger.info(f"mesh: after denoise = {len(pcd.points)}")

    # =====================================================
    # NORMALS (CRITICAL SECTION)
    # =====================================================
    logger.info("mesh: estimating normals")

    bbox = pcd.get_axis_aligned_bounding_box()
    scale = np.linalg.norm(bbox.get_extent())
    radius = scale * 0.006

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius,
            max_nn=60
        )
    )

    # Local consistency
    pcd.orient_normals_consistent_tangent_plane(50)

    # 🔥 GLOBAL FIX (THIS WAS MISSING)
    pcd = _enforce_outward_normals(pcd, logger)

    # =====================================================
    # SCALE ESTIMATION
    # =====================================================
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)

    # =====================================================
    # POISSON
    # =====================================================
    pts = len(pcd.points)

    if pts > 1_000_000:
        depth = 12
    elif pts > 400_000:
        depth = 11
    else:
        depth = 10

    logger.info(f"mesh: poisson depth = {depth}")

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=depth,
        scale=1.03,
        linear_fit=True
    )

    densities = np.asarray(densities)

    # Stronger density filtering
    thresh = np.quantile(densities, 0.02)
    mesh.remove_vertices_by_mask(densities < thresh)

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()

    logger.info(f"mesh: poisson vertices = {len(mesh.vertices)}")

    # =====================================================
    # TRIM FLOATING SHELL
    # =====================================================
    logger.info("mesh: trimming unsupported regions")

    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    verts = np.asarray(mesh.vertices)

    keep = []
    max_dist = avg_dist * 2.0

    for i, v in enumerate(verts):
        _, _, d = pcd_tree.search_knn_vector_3d(v, 1)
        if len(d) > 0 and d[0] < max_dist:
            keep.append(i)

    mesh = mesh.select_by_index(keep)
    mesh.remove_unreferenced_vertices()

    # =====================================================
    # BPA REFINEMENT
    # =====================================================
    logger.info("mesh: BPA refinement")

    radii = [
        avg_dist * 1.5,
        avg_dist * 2.0,
        avg_dist * 2.5
    ]

    mesh_bpa = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd,
        o3d.utility.DoubleVector(radii)
    )

    mesh_bpa.remove_degenerate_triangles()
    mesh_bpa.remove_duplicated_triangles()
    mesh_bpa.remove_non_manifold_edges()

    # =====================================================
    # FILTER BPA
    # =====================================================
    logger.info("mesh: filtering BPA")

    poisson_pts = np.asarray(mesh.vertices)

    poisson_pcd = o3d.geometry.PointCloud()
    poisson_pcd.points = o3d.utility.Vector3dVector(poisson_pts)

    kdtree = o3d.geometry.KDTreeFlann(poisson_pcd)

    keep = []
    threshold = avg_dist * 1.5

    for i, v in enumerate(np.asarray(mesh_bpa.vertices)):
        _, _, d = kdtree.search_knn_vector_3d(v, 1)
        if len(d) > 0 and d[0] < threshold:
            keep.append(i)

    mesh_bpa = mesh_bpa.select_by_index(keep)
    mesh_bpa.remove_unreferenced_vertices()

    # =====================================================
    # MERGE
    # =====================================================
    mesh += mesh_bpa
    mesh.remove_duplicated_vertices()

    # =====================================================
    # REMOVE FLOATING CLUSTERS
    # =====================================================
    clusters, counts, _ = mesh.cluster_connected_triangles()
    counts = np.array(counts)

    largest = counts.max()
    keep = counts > (0.02 * largest)

    remove_mask = [not keep[c] for c in clusters]
    mesh.remove_triangles_by_mask(remove_mask)
    mesh.remove_unreferenced_vertices()

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()

    # =====================================================
    # FINAL NORMALS (MESH)
    # =====================================================
    mesh.compute_vertex_normals()

    # 🔥 ensure final mesh normals consistent too
    mesh.orient_triangles()

    # =====================================================
    # SAVE
    # =====================================================
    o3d.io.write_triangle_mesh(str(output_mesh), mesh)

    logger.info(f"{stage}: SUCCESS → {output_mesh}")