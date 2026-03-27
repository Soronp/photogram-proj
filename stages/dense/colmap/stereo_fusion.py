from pathlib import Path
import open3d as o3d
import numpy as np


# =====================================================
# VALIDATE DENSE WORKSPACE
# =====================================================
def _validate_dense_workspace(dense: Path):
    stereo = dense / "stereo"
    depth = stereo / "depth_maps"
    normal = stereo / "normal_maps"

    return depth.exists() and normal.exists()


# =====================================================
# PARAM BUILDER (HIGH DENSITY, NOT DESTRUCTIVE)
# =====================================================
def _build_params(num_images):
    """
    Goal:
    - maximize point count
    - keep structure stable
    - avoid over-pruning
    """

    if num_images < 40:
        max_pixels = 20000
    elif num_images < 100:
        max_pixels = 30000
    else:
        max_pixels = 40000

    return {
        "max_image_size": -1,

        # 🔥 CRITICAL: allow more growth
        "min_num_pixels": 2,
        "max_num_pixels": max_pixels,

        # 🔥 RELAXED (keeps surfaces alive)
        "max_reproj_error": 3.0,
        "max_depth_error": 0.02,
        "max_normal_error": 25,

        "max_traversal_depth": 150,

        # 🔥 HUGE IMPACT (default=50 is TOO STRICT)
        "check_num_images": 3,

        "cache_size": 64,
        "use_cache": 1,
    }


# =====================================================
# BUILD COMMAND (STRICTLY MATCH CLI)
# =====================================================
def _build_cmd(dense_dir, out_path, p):
    return [
        "colmap", "stereo_fusion",

        "--workspace_path", str(dense_dir),
        "--workspace_format", "COLMAP",

        "--input_type", "geometric",
        "--output_type", "PLY",
        "--output_path", str(out_path),

        "--StereoFusion.max_image_size", str(p["max_image_size"]),

        "--StereoFusion.min_num_pixels", str(p["min_num_pixels"]),
        "--StereoFusion.max_num_pixels", str(p["max_num_pixels"]),

        "--StereoFusion.max_traversal_depth",
        str(p["max_traversal_depth"]),

        "--StereoFusion.max_reproj_error",
        str(p["max_reproj_error"]),

        "--StereoFusion.max_depth_error",
        str(p["max_depth_error"]),

        "--StereoFusion.max_normal_error",
        str(p["max_normal_error"]),

        "--StereoFusion.check_num_images",
        str(p["check_num_images"]),

        "--StereoFusion.cache_size",
        str(p["cache_size"]),

        "--StereoFusion.use_cache",
        str(p["use_cache"]),
    ]


# =====================================================
# LOAD POINT CLOUD
# =====================================================
def _load_pcd(path):
    if not path.exists():
        return None

    pcd = o3d.io.read_point_cloud(str(path))
    if len(pcd.points) == 0:
        return None

    return pcd


# =====================================================
# LIGHT POST PROCESS (NON-DESTRUCTIVE)
# =====================================================
def _post_process(pcd):
    if pcd is None:
        return None

    pts = np.asarray(pcd.points)

    # --- only downsample if EXTREMELY large
    if len(pts) > 1_000_000:
        pcd = pcd.voxel_down_sample(voxel_size=0.005)

    # --- VERY light cleanup (keep density)
    filtered, _ = pcd.remove_statistical_outlier(
        nb_neighbors=8,
        std_ratio=3.0
    )

    # keep if not destructive
    if len(filtered.points) > 0.85 * len(pcd.points):
        pcd = filtered

    return pcd


# =====================================================
# MAIN
# =====================================================
def run(paths, config, logger, tool_runner):
    stage = "stereo_fusion_single_pass"
    logger.info(f"==== {stage.upper()} ====")

    dense_dir = paths.dense

    # -------------------------------------------------
    # VALIDATION
    # -------------------------------------------------
    if not _validate_dense_workspace(dense_dir):
        raise RuntimeError(
            "Dense workspace invalid → run patch_match_stereo first"
        )

    # -------------------------------------------------
    # DATASET SIZE
    # -------------------------------------------------
    num_images = len(list(paths.images.glob("*")))
    logger.info(f"Images detected: {num_images}")

    # -------------------------------------------------
    # PARAM BUILD
    # -------------------------------------------------
    params = _build_params(num_images)
    logger.info(f"Params: {params}")

    # -------------------------------------------------
    # RUN FUSION
    # -------------------------------------------------
    out_path = dense_dir / "fused.ply"

    cmd = _build_cmd(dense_dir, out_path, params)

    tool_runner.run(cmd, stage=stage)

    # -------------------------------------------------
    # LOAD RESULT
    # -------------------------------------------------
    pcd = _load_pcd(out_path)

    if pcd is None:
        raise RuntimeError("Fusion failed → empty output")

    logger.info(f"Raw points: {len(pcd.points)}")

    # -------------------------------------------------
    # POST PROCESS
    # -------------------------------------------------
    pcd = _post_process(pcd)

    logger.info(f"Final points: {len(pcd.points)}")

    # -------------------------------------------------
    # SAVE FINAL
    # -------------------------------------------------
    final_path = dense_dir / "fused_final.ply"
    o3d.io.write_point_cloud(str(final_path), pcd)

    print("\n=== FUSION RESULT ===")
    print(f"Points: {len(pcd.points)}")

    if len(pcd.points) > 500000:
        print("Quality: HIGH")
    elif len(pcd.points) > 100000:
        print("Quality: MEDIUM")
    else:
        print("Quality: LOW")

    return {
        "status": "complete",
        "points": len(pcd.points),
        "images": num_images
    }