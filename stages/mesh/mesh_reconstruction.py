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

    # 🔥 ALWAYS REBUILD (no skipping in adaptive system)
    if output_mesh.exists():
        logger.warning(f"{stage}: removing previous mesh")
        output_mesh.unlink()

    # =====================================================
    # LOAD POINT CLOUD
    # =====================================================
    logger.info(f"{stage}: loading point cloud...")
    pcd = o3d.io.read_point_cloud(str(input_ply))

    num_points = len(pcd.points)

    if num_points < 100:
        raise RuntimeError(f"{stage}: point cloud too small")

    logger.info(f"{stage}: input points = {num_points}")

    # =====================================================
    # 🔥 ADAPTIVE VOXEL SIZE (CRITICAL FIX)
    # =====================================================
    if num_points > 800000:
        voxel_size = 0.003
    elif num_points > 400000:
        voxel_size = 0.004
    else:
        voxel_size = 0.005

    logger.info(f"{stage}: voxel downsampling = {voxel_size}")
    pcd = pcd.voxel_down_sample(voxel_size)

    # =====================================================
    # OUTLIER REMOVAL (LESS AGGRESSIVE)
    # =====================================================
    logger.info(f"{stage}: removing outliers...")
    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=20,
        std_ratio=3.0
    )

    # =====================================================
    # NORMAL ESTIMATION
    # =====================================================
    logger.info(f"{stage}: estimating normals...")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 6,
            max_nn=60
        )
    )
    pcd.orient_normals_consistent_tangent_plane(100)

    # =====================================================
    # 🔥 ADAPTIVE POISSON DEPTH (KEY UPGRADE)
    # =====================================================
    if num_points > 800000:
        depth = 12
    elif num_points > 400000:
        depth = 11
    else:
        depth = 10

    logger.info(f"{stage}: poisson reconstruction (depth={depth})")

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=depth,
        scale=1.1,
        linear_fit=False
    )

    # =====================================================
    # 🔥 MINIMAL DENSITY FILTERING
    # =====================================================
    densities = np.asarray(densities)

    if num_points > 500000:
        threshold = np.quantile(densities, 0.005)  # preserve detail
    else:
        threshold = np.quantile(densities, 0.01)

    mesh.remove_vertices_by_mask(densities < threshold)

    # =====================================================
    # KEEP LARGEST COMPONENT
    # =====================================================
    logger.info(f"{stage}: removing small components...")
    labels = np.array(mesh.cluster_connected_triangles()[0])
    largest_cluster = np.argmax(np.bincount(labels))
    mesh.remove_triangles_by_mask(labels != largest_cluster)

    # =====================================================
    # FINAL CLEANUP
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