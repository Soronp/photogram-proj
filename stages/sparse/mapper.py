from pathlib import Path
import shutil


# =====================================================
# VALIDATION
# =====================================================
def _validate_model(model_path: Path):
    required = ["cameras.bin", "images.bin", "points3D.bin"]
    return all((model_path / f).exists() for f in required)


# =====================================================
# PARAMETER ADAPTATION (CORE LOGIC)
# =====================================================
def _get_adaptive_params(retry: int):
    """
    Returns mapper parameters adapted for retries.
    Focus: maximize coverage while maintaining stability.
    """

    params = {
        "init_inliers": 40,
        "abs_inliers": 30,
        "abs_ratio": 0.25,
        "tri_angle": 1.0,
        "tri_reproj": 6.0,
        "filter_reproj": 4.5,
    }

    if retry > 0:
        # 🔥 Controlled relaxation (NOT explosion)
        params["init_inliers"] = max(20, params["init_inliers"] - 10 * retry)
        params["abs_inliers"] = max(15, params["abs_inliers"] - 5 * retry)

        params["abs_ratio"] = max(0.15, params["abs_ratio"] - 0.05 * retry)

        params["tri_angle"] = max(0.6, params["tri_angle"] - 0.2 * retry)
        params["tri_reproj"] = min(9.0, params["tri_reproj"] + 1.0 * retry)

        params["filter_reproj"] = min(6.0, params["filter_reproj"] + 0.5 * retry)

    return params


# =====================================================
# COLMAP COMMAND BUILDER
# =====================================================
def _build_colmap_cmd(database_path, image_dir, sparse_root, params, use_gpu=True):
    gpu_flag = "1" if use_gpu else "0"
    gpu_index = "0" if use_gpu else "-1"

    return [
        "colmap", "mapper",

        "--database_path", str(database_path),
        "--image_path", str(image_dir),
        "--output_path", str(sparse_root),

        # Threading
        "--Mapper.num_threads", "-1",

        # Model control
        "--Mapper.multiple_models", "0",   # 🔥 force single global model
        "--Mapper.max_num_models", "1",
        "--Mapper.min_model_size", "8",

        # Initialization
        "--Mapper.init_min_num_inliers", str(params["init_inliers"]),
        "--Mapper.init_max_reg_trials", "8",
        "--Mapper.init_num_trials", "500",

        # Pose estimation (CRITICAL)
        "--Mapper.abs_pose_min_num_inliers", str(params["abs_inliers"]),
        "--Mapper.abs_pose_min_inlier_ratio", str(params["abs_ratio"]),
        "--Mapper.abs_pose_max_error", "12",

        # 🔥 Triangulation (KEY FOR FULL OBJECT)
        "--Mapper.tri_min_angle", str(params["tri_angle"]),
        "--Mapper.tri_complete_max_reproj_error", str(params["tri_reproj"]),
        "--Mapper.tri_merge_max_reproj_error", str(params["tri_reproj"]),
        "--Mapper.tri_continue_max_angle_error", "4",

        # 🔥 KEEP ALL TRACKS (CRITICAL FIX)
        "--Mapper.tri_ignore_two_view_tracks", "0",

        # 🔥 Prevent over-pruning
        "--Mapper.filter_max_reproj_error", str(params["filter_reproj"]),

        # 🔥 Improve completeness
        "--Mapper.ba_refine_focal_length", "1",
        "--Mapper.ba_refine_principal_point", "0",
        "--Mapper.ba_refine_extra_params", "1",

        # Bundle Adjustment
        "--Mapper.ba_local_max_num_iterations", "25",
        "--Mapper.ba_global_max_num_iterations", "60",

        "--Mapper.ba_use_gpu", gpu_flag,
        "--Mapper.ba_gpu_index", gpu_index,
    ]


# =====================================================
# GLOMAP COMMAND BUILDER
# =====================================================
def _build_glomap_cmd(database_path, image_dir, sparse_root, retry, use_gpu=True):
    gpu_flag = "1" if use_gpu else "0"
    gpu_index = "0" if use_gpu else "-1"

    # Slight adaptation for retry
    min_tracks = 30 if retry == 0 else 20
    min_views = 3 if retry == 0 else 2

    return [
        "glomap", "mapper",

        "--database_path", str(database_path),
        "--image_path", str(image_dir),
        "--output_path", str(sparse_root),

        "--constraint_type", "POINTS_AND_CAMERAS",

        "--ba_iteration_num", "5",
        "--retriangulation_iteration_num", "3",

        "--RelPoseEstimation.max_epipolar_error", "1",

        "--TrackEstablishment.min_num_tracks_per_view", str(min_tracks),
        "--TrackEstablishment.min_num_view_per_track", str(min_views),

        "--GlobalPositioning.use_gpu", gpu_flag,
        "--GlobalPositioning.gpu_index", gpu_index,

        "--BundleAdjustment.use_gpu", gpu_flag,
        "--BundleAdjustment.gpu_index", gpu_index,

        # 🔥 Relaxed triangulation
        "--Triangulation.min_angle", "0.8",
        "--Triangulation.complete_max_reproj_error", "12",
        "--Triangulation.merge_max_reproj_error", "12",

        "--Thresholds.min_inlier_num", "20",
        "--Thresholds.min_inlier_ratio", "0.2",
        "--Thresholds.max_reprojection_error", "6",
    ]


# =====================================================
# MAIN ENTRY
# =====================================================
def run(paths, config, logger, tool_runner):

    stage = "mapper"
    logger.info(f"---- {stage.upper()} ----")

    image_dir = paths.images
    database_path = paths.database
    sparse_root = paths.sparse

    sparse_root.mkdir(parents=True, exist_ok=True)

    if not database_path.exists():
        raise RuntimeError(f"{stage}: database missing")

    # =====================================================
    # CLEAN PREVIOUS MODELS
    # =====================================================
    logger.info(f"{stage}: clearing previous models")

    for item in sparse_root.iterdir():
        if item.is_dir():
            shutil.rmtree(item)

    backend = config.get("sparse", {}).get("backend", "colmap")
    retry = config.get("_meta", {}).get("retry_count", 0)

    logger.info(f"{stage}: backend={backend}, retry={retry}")

    # =====================================================
    # PARAMETER ADAPTATION
    # =====================================================
    params = _get_adaptive_params(retry)

    logger.info(
        f"{stage}: params | "
        f"init={params['init_inliers']} "
        f"abs={params['abs_inliers']} "
        f"angle={params['tri_angle']} "
        f"reproj={params['tri_reproj']}"
    )

    # =====================================================
    # EXECUTION
    # =====================================================
    try:
        if backend == "glomap":
            cmd = _build_glomap_cmd(database_path, image_dir, sparse_root, retry, True)
            tool_runner.run(cmd, stage=stage + "_glomap_gpu")
        else:
            cmd = _build_colmap_cmd(database_path, image_dir, sparse_root, params, True)
            tool_runner.run(cmd, stage=stage + "_colmap_gpu")

    except Exception as e:
        logger.warning(f"{stage}: GPU failed → {e}")

        if backend == "glomap":
            cmd = _build_glomap_cmd(database_path, image_dir, sparse_root, retry, False)
            tool_runner.run(cmd, stage=stage + "_glomap_cpu")
        else:
            cmd = _build_colmap_cmd(database_path, image_dir, sparse_root, params, False)
            tool_runner.run(cmd, stage=stage + "_colmap_cpu")

    # =====================================================
    # VALIDATION
    # =====================================================
    models = [p for p in sparse_root.iterdir() if p.is_dir()]
    valid_models = [m for m in models if _validate_model(m)]

    if not valid_models:
        raise RuntimeError(f"{stage}: no valid models produced")

    # 🔥 BEST MODEL SELECTION (robust)
    best_model = max(
        valid_models,
        key=lambda m: (m / "points3D.bin").stat().st_size
    )

    logger.info(f"{stage}: selected model → {best_model.name}")

    # 🔥 CRITICAL: enforce canonical path
    if best_model != paths.sparse_model:
        if paths.sparse_model.exists():
            shutil.rmtree(paths.sparse_model)
        shutil.copytree(best_model, paths.sparse_model)

    logger.info(f"{stage}: SUCCESS")