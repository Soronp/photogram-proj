from pathlib import Path
import shutil


# =====================================================
# VALIDATION
# =====================================================
def _validate_model(model_path: Path):
    required = ["cameras.bin", "images.bin", "points3D.bin"]
    return all((model_path / f).exists() for f in required)


# =====================================================
# PARAMETER ADAPTATION (SAFE)
# =====================================================
def _get_params(retry: int):
    """
    Geometry-safe parameter adaptation.
    """

    params = {
        # Initialization (closer to COLMAP defaults)
        "init_inliers": 100,
        "init_tri_angle": 16,
        "init_max_error": 4,

        # Pose
        "abs_inliers": 30,
        "abs_ratio": 0.25,

        # Triangulation (CRITICAL)
        "tri_angle": 1.5,
        "tri_reproj": 4.0,
        "filter_reproj": 4.0,
    }

    if retry > 0:
        # VERY controlled relaxation
        params["init_inliers"] = max(60, params["init_inliers"] - 20 * retry)
        params["abs_inliers"] = max(20, params["abs_inliers"] - 5 * retry)

        params["tri_angle"] = max(1.2, params["tri_angle"] - 0.1 * retry)
        params["tri_reproj"] = min(6.0, params["tri_reproj"] + 0.5 * retry)
        params["filter_reproj"] = min(5.0, params["filter_reproj"] + 0.5 * retry)

    return params


# =====================================================
# COLMAP COMMAND (FULLY CORRECT)
# =====================================================
def _build_colmap_cmd(database_path, image_dir, sparse_root, p, use_gpu=True):

    return [
        "colmap", "mapper",

        "--database_path", str(database_path),
        "--image_path", str(image_dir),
        "--output_path", str(sparse_root),

        # Threading
        "--Mapper.num_threads", "-1",

        # SINGLE MODEL (important)
        "--Mapper.multiple_models", "0",
        "--Mapper.max_num_models", "1",
        "--Mapper.min_model_size", "10",

        # -------------------------
        # INITIALIZATION (FIXED)
        # -------------------------
        "--Mapper.init_min_num_inliers", str(p["init_inliers"]),
        "--Mapper.init_max_error", str(p["init_max_error"]),
        "--Mapper.init_min_tri_angle", str(p["init_tri_angle"]),
        "--Mapper.init_max_reg_trials", "4",
        "--Mapper.init_num_trials", "300",

        # -------------------------
        # POSE
        # -------------------------
        "--Mapper.abs_pose_min_num_inliers", str(p["abs_inliers"]),
        "--Mapper.abs_pose_min_inlier_ratio", str(p["abs_ratio"]),
        "--Mapper.abs_pose_max_error", "12",

        # -------------------------
        # TRIANGULATION (CRITICAL FIX)
        # -------------------------
        "--Mapper.tri_min_angle", str(p["tri_angle"]),
        "--Mapper.tri_complete_max_reproj_error", str(p["tri_reproj"]),
        "--Mapper.tri_merge_max_reproj_error", str(p["tri_reproj"]),
        "--Mapper.tri_create_max_angle_error", "2",
        "--Mapper.tri_continue_max_angle_error", "2",

        # KEEP TRACKS
        "--Mapper.tri_ignore_two_view_tracks", "0",

        # FILTERING
        "--Mapper.filter_max_reproj_error", str(p["filter_reproj"]),
        "--Mapper.filter_min_tri_angle", "1.5",

        # -------------------------
        # BUNDLE ADJUSTMENT
        # -------------------------
        "--Mapper.ba_refine_focal_length", "1",
        "--Mapper.ba_refine_principal_point", "0",
        "--Mapper.ba_refine_extra_params", "1",

        "--Mapper.ba_local_max_num_iterations", "25",
        "--Mapper.ba_global_max_num_iterations", "50",

        "--Mapper.ba_use_gpu", "1" if use_gpu else "0",
        "--Mapper.ba_gpu_index", "0" if use_gpu else "-1",
    ]


# =====================================================
# GLOMAP COMMAND (FIXED SCALE)
# =====================================================
def _build_glomap_cmd(database_path, image_dir, sparse_root, retry, use_gpu=True):

    return [
        "glomap", "mapper",

        "--database_path", str(database_path),
        "--image_path", str(image_dir),
        "--output_path", str(sparse_root),

        "--constraint_type", "POINTS_AND_CAMERAS",

        "--ba_iteration_num", "3",
        "--retriangulation_iteration_num", "2",

        "--RelPoseEstimation.max_epipolar_error", "1",

        "--TrackEstablishment.min_num_tracks_per_view",
        str(30 if retry == 0 else 20),

        "--TrackEstablishment.min_num_view_per_track",
        str(3 if retry == 0 else 2),

        # GPU
        "--GlobalPositioning.use_gpu", "1" if use_gpu else "0",
        "--GlobalPositioning.gpu_index", "0" if use_gpu else "-1",

        "--BundleAdjustment.use_gpu", "1" if use_gpu else "0",
        "--BundleAdjustment.gpu_index", "0" if use_gpu else "-1",

        # -------------------------
        # TRIANGULATION (FIXED)
        # -------------------------
        "--Triangulation.min_angle", "1.0",
        "--Triangulation.complete_max_reproj_error", "15",
        "--Triangulation.merge_max_reproj_error", "15",

        # -------------------------
        # THRESHOLDS (CRITICAL FIX)
        # -------------------------
        "--Thresholds.min_inlier_num", "30",
        "--Thresholds.min_inlier_ratio", "0.25",
        "--Thresholds.max_reprojection_error", "0.01",  # 🔥 FIXED
    ]


# =====================================================
# MAIN
# =====================================================
def run(paths, config, logger, tool_runner):

    stage = "mapper"
    logger.info(f"---- {stage.upper()} ----")

    image_dir = paths.images
    database_path = paths.database
    sparse_root = paths.sparse

    sparse_root.mkdir(parents=True, exist_ok=True)

    if not database_path.exists():
        raise RuntimeError("database missing")

    # =================================================
    # CLEAN
    # =================================================
    for item in sparse_root.iterdir():
        if item.is_dir():
            shutil.rmtree(item)

    backend = config["pipeline"]["backends"]["sparse"]
    retry = config.get("_meta", {}).get("retry_count", 0)

    logger.info(f"{stage}: backend={backend}, retry={retry}")

    params = _get_params(retry)

    # =================================================
    # RUN
    # =================================================
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

    # =================================================
    # VALIDATE
    # =================================================
    models = [p for p in sparse_root.iterdir() if p.is_dir()]
    valid_models = [m for m in models if _validate_model(m)]

    if not valid_models:
        raise RuntimeError("no valid models")

    best_model = max(
        valid_models,
        key=lambda m: (m / "points3D.bin").stat().st_size
    )

    if best_model != paths.sparse_model:
        if paths.sparse_model.exists():
            shutil.rmtree(paths.sparse_model)
        shutil.copytree(best_model, paths.sparse_model)

    logger.info(f"{stage}: SUCCESS → {best_model.name}")