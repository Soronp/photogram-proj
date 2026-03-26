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
# CAMERA LOADING
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
            f.read(4)

            qvec = struct.unpack("<4d", f.read(32))
            tvec = struct.unpack("<3d", f.read(24))

            f.read(4)

            while True:
                if f.read(1) == b"\x00":
                    break

            num_points2D = struct.unpack("<Q", f.read(8))[0]
            f.read(num_points2D * 24)

            R = _qvec_to_rotmat(qvec)
            t = np.array(tvec)

            center = -R.T @ t
            centers.append(center)

    centers = np.array(centers)

    # 🔥 CRITICAL FIX: VALIDATION
    if centers.size == 0:
        return None

    if centers.ndim != 2 or centers.shape[1] != 3:
        return None

    # Remove NaN / Inf
    mask = np.isfinite(centers).all(axis=1)
    centers = centers[mask]

    if len(centers) == 0:
        return None

    return centers


# =====================================================
# SCALE
# =====================================================
def _compute_scale(pcd):
    pts = np.asarray(pcd.points)
    centroid = np.mean(pts, axis=0)
    dists = np.linalg.norm(pts - centroid, axis=1)
    return np.percentile(dists, 90)


# =====================================================
# NORMALS (ROBUST)
# =====================================================
def _compute_normals(pcd, camera_centers, logger):
    logger.info("mesh: computing stable normals")

    scale = _compute_scale(pcd)
    radius = scale * 0.015

    # Step 1: Estimate
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius,
            max_nn=60
        )
    )

    # Step 2: Local consistency
    pcd.orient_normals_consistent_tangent_plane(200)

    # -------------------------------------------------
    # 🔥 SAFE CAMERA-BASED ORIENTATION
    # -------------------------------------------------
    if camera_centers is None or len(camera_centers) < 2:
        logger.warning("mesh: camera centers invalid → fallback to Open3D orientation")
        pcd.orient_normals_towards_camera_location(pcd.get_center())
        return pcd

    try:
        cam_tree = o3d.geometry.KDTreeFlann(
            o3d.utility.Vector3dVector(camera_centers)
        )
    except Exception as e:
        logger.warning(f"mesh: KDTree build failed → {e}")
        pcd.orient_normals_towards_camera_location(pcd.get_center())
        return pcd

    pts = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)

    valid_count = 0

    for i in range(len(pts)):
        try:
            p = pts[i]
            n = normals[i]

            k, idx, _ = cam_tree.search_knn_vector_3d(p, 1)

            if k < 1:
                continue

            cam = camera_centers[idx[0]]
            view_dir = p - cam

            if np.dot(n, view_dir) < 0:
                normals[i] *= -1

            valid_count += 1

        except Exception:
            continue  # 🔥 NEVER CRASH

    if valid_count == 0:
        logger.warning("mesh: camera orientation failed → fallback")
        pcd.orient_normals_towards_camera_location(pcd.get_center())
    else:
        logger.info(f"mesh: oriented {valid_count} points using cameras")

    pcd.normals = o3d.utility.Vector3dVector(normals)

    return pcd


# =====================================================
# POISSON
# =====================================================
def _run_poisson(pcd, logger):
    pts = len(pcd.points)

    depth = 11 if pts > 500_000 else 10
    logger.info(f"mesh: poisson depth = {depth}")

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=depth,
        scale=1.02,
        linear_fit=True
    )

    densities = np.asarray(densities)

    thresh = np.quantile(densities, 0.05)
    mesh.remove_vertices_by_mask(densities < thresh)

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
        try:
            k, _, d = kdtree.search_knn_vector_3d(v, 1)
            if k > 0 and d[0] < avg_dist * 1.5:
                keep.append(i)
        except Exception:
            continue

    mesh = mesh.select_by_index(keep)
    mesh.remove_unreferenced_vertices()

    return mesh


# =====================================================
# CLEAN
# =====================================================
def _clean(mesh):
    clusters, counts, _ = mesh.cluster_connected_triangles()
    counts = np.array(counts)

    keep = counts > (0.05 * counts.max())
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

    logger.info("---- MESH_RECONSTRUCTION (ROBUST FIXED) ----")

    if not paths.fused.exists():
        raise RuntimeError("fused.ply missing")

    images_bin = paths.dense / "sparse" / "images.bin"
    if not images_bin.exists():
        raise RuntimeError("images.bin required")

    pcd = o3d.io.read_point_cloud(str(paths.fused))

    if len(pcd.points) < 5000:
        raise RuntimeError("insufficient points")

    if len(pcd.points) > 1_500_000:
        pcd = pcd.voxel_down_sample(0.003)

    pcd, _ = pcd.remove_statistical_outlier(20, 2.5)

    camera_centers = _load_camera_centers(images_bin)

    pcd = _compute_normals(pcd, camera_centers, logger)

    mesh = _run_poisson(pcd, logger)
    mesh = _trim(mesh, pcd)
    mesh = _clean(mesh)

    mesh.compute_vertex_normals()

    o3d.io.write_triangle_mesh(str(paths.mesh_file), mesh)

    logger.info(f"mesh: SUCCESS → {paths.mesh_file}")