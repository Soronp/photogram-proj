from pathlib import Path
import open3d as o3d
import numpy as np
import struct


# =====================================================
# DETERMINISM
# =====================================================
def _set_determinism():
    np.random.seed(42)


# =====================================================
# COLMAP CAMERA LOADING
# =====================================================
def _qvec_to_rotmat(qvec):
    w, x, y, z = qvec
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])


def _load_camera_centers(images_bin):
    centers = []

    with open(images_bin, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]

        for _ in range(num_images):
            f.read(4)  # image_id

            qvec = struct.unpack("<4d", f.read(32))
            tvec = struct.unpack("<3d", f.read(24))

            f.read(4)  # camera_id

            # read name
            while True:
                if f.read(1) == b"\x00":
                    break

            num_points2D = struct.unpack("<Q", f.read(8))[0]
            f.read(num_points2D * 24)

            R = _qvec_to_rotmat(qvec)
            t = np.array(tvec)

            center = -R.T @ t
            centers.append(center)

    return np.array(centers)


# =====================================================
# SCALE
# =====================================================
def _compute_scale(pcd):
    pts = np.asarray(pcd.points)
    centroid = np.mean(pts, axis=0)
    dists = np.linalg.norm(pts - centroid, axis=1)
    return np.percentile(dists, 90)


# =====================================================
# NORMALS (CAMERA-BASED)
# =====================================================
def _compute_normals(pcd, camera_centers, logger):
    logger.info("mesh: computing camera-aware normals")

    scale = _compute_scale(pcd)
    radius = scale * 0.01

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius,
            max_nn=50
        )
    )

    # local consistency
    pcd.orient_normals_consistent_tangent_plane(100)

    pts = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)

    # 🔥 camera-based orientation
    for i in range(len(pts)):
        p = pts[i]
        n = normals[i]

        votes = 0
        for cam in camera_centers:
            view = p - cam
            votes += 1 if np.dot(n, view) > 0 else -1

        if votes < 0:
            normals[i] *= -1

    pcd.normals = o3d.utility.Vector3dVector(normals)

    return pcd, scale


# =====================================================
# POISSON
# =====================================================
def _run_poisson(pcd, logger):
    pts = len(pcd.points)

    depth = 12 if pts > 1_000_000 else 11 if pts > 400_000 else 10
    logger.info(f"mesh: poisson depth = {depth}")

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=depth,
        scale=1.03,
        linear_fit=True
    )

    densities = np.asarray(densities)
    thresh = np.quantile(densities, 0.02)
    mesh.remove_vertices_by_mask(densities < thresh)

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()

    return mesh


# =====================================================
# TRIM
# =====================================================
def _trim(mesh, pcd):
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)

    kdtree = o3d.geometry.KDTreeFlann(pcd)
    verts = np.asarray(mesh.vertices)

    keep = []
    for i, v in enumerate(verts):
        _, _, d = kdtree.search_knn_vector_3d(v, 1)
        if len(d) and d[0] < avg_dist * 2.0:
            keep.append(i)

    mesh = mesh.select_by_index(keep)
    mesh.remove_unreferenced_vertices()

    return mesh, avg_dist


# =====================================================
# BPA (STABLE)
# =====================================================
def _run_bpa(pcd, avg_dist):
    radii = [avg_dist * 1.5, avg_dist * 2.0]

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd,
        o3d.utility.DoubleVector(radii)
    )

    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()

    return mesh


# =====================================================
# CLEAN
# =====================================================
def _clean(mesh):
    clusters, counts, _ = mesh.cluster_connected_triangles()
    counts = np.array(counts)

    keep = counts > (0.02 * counts.max())
    mask = [not keep[c] for c in clusters]

    mesh.remove_triangles_by_mask(mask)
    mesh.remove_unreferenced_vertices()

    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()

    return mesh


# =====================================================
# MAIN
# =====================================================
def run(paths, config, logger):
    _set_determinism()

    logger.info("---- MESH_RECONSTRUCTION ----")

    if not paths.fused.exists():
        raise RuntimeError("fused.ply missing")

    images_bin = paths.dense / "sparse" / "images.bin"
    if not images_bin.exists():
        raise RuntimeError("images.bin required for stable normals")

    logger.info("mesh: loading data")

    pcd = o3d.io.read_point_cloud(str(paths.fused))

    if len(pcd.points) < 5000:
        raise RuntimeError("insufficient points")

    # downsample
    if len(pcd.points) > 1_500_000:
        pcd = pcd.voxel_down_sample(0.0025)

    # denoise
    pcd, _ = pcd.remove_statistical_outlier(16, 3.0)

    # load cameras
    camera_centers = _load_camera_centers(images_bin)

    # normals
    pcd, _ = _compute_normals(pcd, camera_centers, logger)

    # poisson
    mesh = _run_poisson(pcd, logger)

    # trim
    mesh, avg_dist = _trim(mesh, pcd)

    # BPA (reduced influence)
    mesh_bpa = _run_bpa(pcd, avg_dist)

    mesh += mesh_bpa
    mesh.remove_duplicated_vertices()

    # clean
    mesh = _clean(mesh)

    # final normals
    mesh.compute_vertex_normals()
    mesh.orient_triangles()

    # save
    o3d.io.write_triangle_mesh(str(paths.mesh_file), mesh)

    logger.info(f"mesh: SUCCESS → {paths.mesh_file}")