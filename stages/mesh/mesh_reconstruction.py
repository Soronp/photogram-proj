from pathlib import Path
import open3d as o3d
import numpy as np


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
    # LOAD POINT CLOUD
    # =====================================================
    logger.info(f"{stage}: loading fused cloud...")
    pcd = o3d.io.read_point_cloud(str(input_ply))
    num_points = len(pcd.points)

    if num_points < 1000:
        raise RuntimeError(f"{stage}: insufficient points")

    logger.info(f"{stage}: input points = {num_points}")

    # =====================================================
    # 🎯 TARGET POINT BUDGET (CRITICAL)
    # =====================================================
    TARGET_POINTS = 1_000_000

    if num_points > 1_500_000:
        ratio = TARGET_POINTS / num_points
        voxel_size = 0.002 + (0.004 * (1 - ratio))
        logger.info(f"{stage}: downsampling to control density → voxel={voxel_size:.5f}")
        pcd = pcd.voxel_down_sample(voxel_size)
    else:
        voxel_size = 0.003
        logger.info(f"{stage}: light voxel downsampling → {voxel_size}")
        pcd = pcd.voxel_down_sample(voxel_size)

    logger.info(f"{stage}: after voxel = {len(pcd.points)} points")

    # =====================================================
    # 🔥 ROBUST OUTLIER REMOVAL (NOT TOO AGGRESSIVE)
    # =====================================================
    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=20,
        std_ratio=2.5
    )

    logger.info(f"{stage}: after denoise = {len(pcd.points)} points")

    # =====================================================
    # NORMAL ESTIMATION (STABLE)
    # =====================================================
    logger.info(f"{stage}: estimating normals...")

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 5,
            max_nn=50
        )
    )

    pcd.orient_normals_consistent_tangent_plane(50)

    # =====================================================
    # 🎯 POISSON DEPTH CONTROL (KEY FIX)
    # =====================================================
    pts = len(pcd.points)

    if pts > 1_200_000:
        depth = 11
    elif pts > 700_000:
        depth = 10
    else:
        depth = 9

    logger.info(f"{stage}: poisson depth = {depth}")

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=depth,
        scale=1.05,
        linear_fit=True
    )

    # =====================================================
    # 🔥 DENSITY FILTERING (BALANCED)
    # =====================================================
    densities = np.asarray(densities)

    # Keep ~99% for high detail
    threshold = np.quantile(densities, 0.01)

    mesh.remove_vertices_by_mask(densities < threshold)

    # =====================================================
    # KEEP LARGEST COMPONENT ONLY
    # =====================================================
    labels = np.array(mesh.cluster_connected_triangles()[0])
    largest_cluster = np.argmax(np.bincount(labels))

    mesh.remove_triangles_by_mask(labels != largest_cluster)

    # =====================================================
    # CLEANUP
    # =====================================================
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()

    # =====================================================
    # SAVE
    # =====================================================
    o3d.io.write_triangle_mesh(str(output_mesh), mesh)

    logger.info(f"{stage}: SUCCESS → {output_mesh}")