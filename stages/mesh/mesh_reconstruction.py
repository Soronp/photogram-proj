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

    if num_points < 5000:
        raise RuntimeError(f"{stage}: insufficient points")

    logger.info(f"{stage}: input points = {num_points}")

    # =====================================================
    # 🔥 MINIMAL DOWNSAMPLING (PRESERVE DETAIL)
    # =====================================================
    if num_points > 1_500_000:
        voxel_size = 0.0025
        logger.info(f"{stage}: controlled downsample → voxel={voxel_size}")
        pcd = pcd.voxel_down_sample(voxel_size)
    else:
        logger.info(f"{stage}: no downsampling (preserving detail)")

    logger.info(f"{stage}: after voxel = {len(pcd.points)}")

    # =====================================================
    # 🔥 LIGHT DENOISING (SAFE)
    # =====================================================
    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=16,
        std_ratio=3.0   # 🔥 more permissive
    )

    logger.info(f"{stage}: after denoise = {len(pcd.points)}")

    # =====================================================
    # NORMAL ESTIMATION (CRITICAL FOR POISSON)
    # =====================================================
    logger.info(f"{stage}: estimating normals...")

    bbox = pcd.get_axis_aligned_bounding_box()
    scale = np.linalg.norm(bbox.get_extent())

    radius = scale * 0.01  # adaptive radius

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius,
            max_nn=60
        )
    )

    pcd.orient_normals_consistent_tangent_plane(50)

    # =====================================================
    # 🔥 POISSON RECONSTRUCTION (HIGH QUALITY)
    # =====================================================
    pts = len(pcd.points)

    if pts > 1_000_000:
        depth = 12
    elif pts > 400_000:
        depth = 11
    else:
        depth = 10

    logger.info(f"{stage}: poisson depth = {depth}")

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=depth,
        scale=1.1,        # 🔥 slightly expanded → fills holes
        linear_fit=False  # 🔥 sharper detail
    )

    # =====================================================
    # 🔥 VERY LIGHT DENSITY FILTERING (ANTI-HOLE)
    # =====================================================
    densities = np.asarray(densities)

    # 🔥 keep 99.8% → almost everything
    threshold = np.quantile(densities, 0.002)

    logger.info(f"{stage}: density threshold = {threshold:.6f}")

    mesh.remove_vertices_by_mask(densities < threshold)

    # =====================================================
    # 🔥 OPTIONAL: KEEP MULTIPLE COMPONENTS (IMPORTANT)
    # =====================================================
    triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()

    cluster_n_triangles = np.array(cluster_n_triangles)

    # 🔥 keep clusters ≥5% of largest
    largest = cluster_n_triangles.max()
    keep = cluster_n_triangles > (0.05 * largest)

    remove_mask = [not keep[c] for c in triangle_clusters]

    mesh.remove_triangles_by_mask(remove_mask)

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