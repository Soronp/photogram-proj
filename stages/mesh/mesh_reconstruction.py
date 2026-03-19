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
        logger.warning(f"{stage}: mesh already exists, skipping...")
        return

    # -----------------------------
    # Load point cloud
    # -----------------------------
    logger.info(f"{stage}: loading point cloud...")
    pcd = o3d.io.read_point_cloud(str(input_ply))

    if len(pcd.points) < 100:
        raise RuntimeError(f"{stage}: point cloud too small")

    # -----------------------------
    # Downsample (important for stability)
    # -----------------------------
    voxel_size = config.get("mesh", {}).get("voxel_size", 0.005)
    logger.info(f"{stage}: voxel downsampling ({voxel_size})")
    pcd = pcd.voxel_down_sample(voxel_size)

    # -----------------------------
    # Outlier removal (less aggressive)
    # -----------------------------
    logger.info(f"{stage}: removing outliers...")
    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=30,
        std_ratio=2.5
    )

    # -----------------------------
    # Normal estimation (CRITICAL)
    # -----------------------------
    logger.info(f"{stage}: estimating normals...")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 5,
            max_nn=50
        )
    )
    pcd.orient_normals_consistent_tangent_plane(100)

    # -----------------------------
    # Poisson Reconstruction (HIGH QUALITY)
    # -----------------------------
    depth = config.get("mesh", {}).get("poisson_depth", 11)

    logger.info(f"{stage}: poisson reconstruction (depth={depth})")

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=depth,
        scale=1.1,
        linear_fit=False
    )

    # -----------------------------
    # Density-based cleanup (LESS aggressive)
    # -----------------------------
    densities = np.asarray(densities)
    threshold = np.quantile(densities, 0.01)  # 🔥 reduced filtering
    mesh.remove_vertices_by_mask(densities < threshold)

    # -----------------------------
    # Keep largest connected component
    # -----------------------------
    logger.info(f"{stage}: removing small components...")
    labels = np.array(mesh.cluster_connected_triangles()[0])
    largest_cluster = np.argmax(np.bincount(labels))
    triangles_to_keep = labels == largest_cluster
    mesh.remove_triangles_by_mask(~triangles_to_keep)

    # -----------------------------
    # Final cleanup
    # -----------------------------
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()

    # -----------------------------
    # Save
    # -----------------------------
    o3d.io.write_triangle_mesh(str(output_mesh), mesh)

    logger.info(f"{stage}: mesh saved at {output_mesh}")