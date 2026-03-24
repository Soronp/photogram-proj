from pathlib import Path
import open3d as o3d
import numpy as np


# =====================================================
# SAFE LOAD
# =====================================================
def _safe_load_pcd(path: Path, logger, name: str):
    if not path.exists():
        logger.warning(f"{name} missing: {path}")
        return None

    pcd = o3d.io.read_point_cloud(str(path))
    if len(pcd.points) == 0:
        logger.warning(f"{name} is empty")
        return None

    logger.info(f"{name} points = {len(pcd.points)}")
    return pcd


# =====================================================
# SCALE & DENSITY
# =====================================================
def _estimate_scale(pcd):
    bbox = pcd.get_axis_aligned_bounding_box()
    return np.linalg.norm(bbox.get_extent())


def _compute_density(pcd, sample_size=5000):
    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        return 0.001

    sample = pts[np.random.choice(len(pts), min(sample_size, len(pts)), replace=False)]
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    dists = []

    for p in sample:
        _, _, dist = kdtree.search_knn_vector_3d(p, 2)
        if len(dist) > 1:
            dists.append(np.sqrt(dist[1]))

    return np.median(dists) if dists else 0.001


# =====================================================
# SMART MERGE (NO EDGE DESTRUCTION)
# =====================================================
def _merge_pointclouds(pcd_strict, pcd_relaxed, scale, logger):
    logger.info("fusion: smart merge (gap-filling mode)")

    strict_pts = np.asarray(pcd_strict.points)
    relaxed_pts = np.asarray(pcd_relaxed.points)

    kdtree = o3d.geometry.KDTreeFlann(pcd_strict)

    # Much looser threshold → only reject extreme outliers
    dist_thresh = scale * 0.01

    keep = []
    for i, p in enumerate(relaxed_pts):
        _, _, dist = kdtree.search_knn_vector_3d(p, 1)

        # Keep if:
        # ✔ fills gap (far from strict)
        # ✔ OR reasonably close (not an outlier)
        if len(dist) == 0 or dist[0] > dist_thresh:
            keep.append(i)
        else:
            # Also keep close points → improves density
            keep.append(i)

    logger.info(f"relaxed kept = {len(keep)} / {len(relaxed_pts)}")

    relaxed_filtered = pcd_relaxed.select_by_index(keep)

    merged = pcd_strict + relaxed_filtered
    return merged


# =====================================================
# EDGE SAFE POSTPROCESS
# =====================================================
def _edge_preserving_postprocess(pcd, scale, logger):
    logger.info("fusion: edge-safe postprocess")

    density = _compute_density(pcd)

    # 🔥 CRITICAL FIX: smaller voxel
    voxel_size = density * 0.4
    logger.info(f"voxel size = {voxel_size:.6f}")

    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)

    # 🔥 less aggressive outlier removal
    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=20,
        std_ratio=2.5
    )

    # Normals (important for mesh)
    radius = density * 5
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius,
            max_nn=80
        )
    )

    pcd.orient_normals_consistent_tangent_plane(100)

    return pcd


# =====================================================
# RUN COLMAP FUSION
# =====================================================
def _run_fusion(tool_runner, dense_dir, output_path, params, stage_name):
    cmd = [
        "colmap", "stereo_fusion",
        "--workspace_path", str(dense_dir),
        "--workspace_format", "COLMAP",
        "--input_type", "geometric",
        "--output_type", "PLY",
        "--output_path", str(output_path),

        "--StereoFusion.min_num_pixels", str(params["min_pixels"]),
        "--StereoFusion.max_reproj_error", str(params["max_reproj"]),
        "--StereoFusion.max_depth_error", str(params["max_depth"]),
        "--StereoFusion.max_normal_error", str(params["max_normal"]),
        "--StereoFusion.max_traversal_depth", str(params["traversal"]),

        "--StereoFusion.num_threads", "-1",
    ]

    tool_runner.run(cmd, stage=stage_name)


# =====================================================
# MAIN
# =====================================================
def run(paths, config, logger, tool_runner):
    stage = "stereo_fusion"
    logger.info(f"---- {stage.upper()} ----")

    dense_dir = paths.dense
    fused_path = dense_dir / "fused.ply"
    strict_path = dense_dir / "fused_strict.ply"
    relaxed_path = dense_dir / "fused_relaxed.ply"

    # -------------------------------------------------
    # STRICT (structure)
    # -------------------------------------------------
    strict_params = {
        "min_pixels": 2,
        "max_reproj": 2.5,
        "max_depth": 0.02,
        "max_normal": 15,
        "traversal": 100,
    }

    logger.info("PASS 1: STRICT")
    _run_fusion(tool_runner, dense_dir, strict_path, strict_params, stage + "_strict")

    # -------------------------------------------------
    # RELAXED (gap filler)
    # -------------------------------------------------
    relaxed_params = {
        "min_pixels": 1,
        "max_reproj": 4.0,
        "max_depth": 0.05,
        "max_normal": 35,
        "traversal": 200,
    }

    logger.info("PASS 2: RELAXED")
    _run_fusion(tool_runner, dense_dir, relaxed_path, relaxed_params, stage + "_relaxed")

    # -------------------------------------------------
    # LOAD
    # -------------------------------------------------
    pcd_strict = _safe_load_pcd(strict_path, logger, "strict")
    pcd_relaxed = _safe_load_pcd(relaxed_path, logger, "relaxed")

    if pcd_strict is None and pcd_relaxed is None:
        raise RuntimeError("Fusion failed completely")

    # -------------------------------------------------
    # MERGE
    # -------------------------------------------------
    if pcd_strict is not None and pcd_relaxed is not None:
        scale = _estimate_scale(pcd_strict)
        merged = _merge_pointclouds(pcd_strict, pcd_relaxed, scale, logger)

    elif pcd_strict is not None:
        logger.warning("relaxed missing → using strict only")
        scale = _estimate_scale(pcd_strict)
        merged = pcd_strict

    else:
        logger.warning("strict missing → using relaxed only")
        scale = _estimate_scale(pcd_relaxed)
        merged = pcd_relaxed

    # -------------------------------------------------
    # POSTPROCESS
    # -------------------------------------------------
    merged = _edge_preserving_postprocess(merged, scale, logger)

    # -------------------------------------------------
    # SAVE
    # -------------------------------------------------
    o3d.io.write_point_cloud(str(fused_path), merged)

    logger.info(f"{stage}: final points = {len(merged.points)}")
    logger.info(f"{stage}: SUCCESS")

    return {
        "num_points": int(len(merged.points)),
        "status": "stable_edge_preserving"
    }