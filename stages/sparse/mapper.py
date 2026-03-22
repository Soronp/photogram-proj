from pathlib import Path
import shutil


def _validate_model(model_path: Path):
    required = ["cameras.bin", "images.bin", "points3D.bin"]
    return all((model_path / f).exists() for f in required)


def run(paths, config, logger, tool_runner):
    stage = "mapper"
    logger.info(f"---- {stage.upper()} ----")

    image_dir = paths.images
    database_path = paths.database
    sparse_root = paths.sparse
    sparse_root.mkdir(parents=True, exist_ok=True)

    if not database_path.exists():
        raise RuntimeError(f"{stage}: database missing")

    # ----------------------------------------
    # Clean old models ONLY
    # ----------------------------------------
    for item in sparse_root.iterdir():
        if item.is_dir():
            shutil.rmtree(item)

    # ----------------------------------------
    # Analysis
    # ----------------------------------------
    analysis = config.get("analysis_results", {})
    matches = analysis.get("matches", {})

    connectivity = matches.get("connectivity", 0.3)
    retry = config.get("_meta", {}).get("retry_count", 0)

    logger.info(f"{stage}: connectivity={connectivity:.3f}, retry={retry}")

    # =========================================
    # 🔥 COVERAGE-FIRST PARAMS
    # =========================================

    # --- Core inliers ---
    init_inliers = 60
    abs_inliers = 40
    min_model_size = 10

    # --- Expansion ---
    max_reg_trials = 4
    init_max_trials = 200

    # --- Triangulation (CRITICAL FOR FULL SHAPE) ---
    tri_min_angle = 1.0
    tri_complete_max_reproj = 6.0
    tri_merge_max_reproj = 6.0
    tri_continue_max_angle = 3.0

    # =========================================
    # 🔥 ADAPTATION
    # =========================================

    if connectivity < 0.2:
        init_inliers = 30
        abs_inliers = 20
        min_model_size = 5
        tri_min_angle = 0.8

    elif connectivity > 0.5:
        init_inliers = 80
        abs_inliers = 60
        tri_min_angle = 1.5

    if retry > 0:
        init_inliers = max(25, int(init_inliers * 0.8))
        abs_inliers = max(20, int(abs_inliers * 0.8))
        max_reg_trials += retry
        tri_complete_max_reproj += 1.0 * retry

    logger.info(
        f"{stage}: inliers={init_inliers}/{abs_inliers}, "
        f"tri_angle={tri_min_angle}, reg_trials={max_reg_trials}"
    )

    # =========================================
    # 🔥 COMMAND BUILDER (VALID FLAGS ONLY)
    # =========================================

    def _build_cmd(use_gpu=True):
        gpu_flag = "1" if use_gpu else "0"
        gpu_index = "0" if use_gpu else "-1"

        return [
            "colmap", "mapper",
            "--database_path", str(database_path),
            "--image_path", str(image_dir),
            "--output_path", str(sparse_root),

            "--Mapper.num_threads", "-1",

            # 🔥 multi-model (prevents collapse)
            "--Mapper.multiple_models", "1",
            "--Mapper.max_num_models", "10",

            # 🔥 initialization
            "--Mapper.init_min_num_inliers", str(init_inliers),
            "--Mapper.init_max_reg_trials", str(max_reg_trials),
            "--Mapper.init_num_trials", str(init_max_trials),

            # 🔥 pose estimation
            "--Mapper.abs_pose_min_num_inliers", str(abs_inliers),

            # 🔥 model size
            "--Mapper.min_model_size", str(min_model_size),

            # 🔥 triangulation (CRITICAL)
            "--Mapper.tri_min_angle", str(tri_min_angle),
            "--Mapper.tri_complete_max_reproj_error", str(tri_complete_max_reproj),
            "--Mapper.tri_merge_max_reproj_error", str(tri_merge_max_reproj),
            "--Mapper.tri_continue_max_angle_error", str(tri_continue_max_angle),

            # 🔥 BA (stable, not restrictive)
            "--Mapper.ba_global_max_num_iterations", "50",
            "--Mapper.ba_local_max_num_iterations", "25",

            "--Mapper.ba_use_gpu", gpu_flag,
            "--Mapper.ba_gpu_index", gpu_index,
        ]

    # =========================================
    # 🔥 RUN
    # =========================================

    try:
        logger.info(f"{stage}: GPU mapping...")
        tool_runner.run(_build_cmd(True), stage=stage + "_gpu")
    except Exception as e:
        logger.warning(f"{stage}: GPU failed → {e}")
        logger.info(f"{stage}: CPU fallback...")
        tool_runner.run(_build_cmd(False), stage=stage + "_cpu")

    # =========================================
    # 🔥 SELECT BEST MODEL
    # =========================================

    models = [p for p in sparse_root.iterdir() if p.is_dir()]

    valid_models = [m for m in models if _validate_model(m)]

    if not valid_models:
        raise RuntimeError(f"{stage}: no valid models produced")

    best_model = max(
        valid_models,
        key=lambda m: (m / "points3D.bin").stat().st_size
    )

    logger.info(f"{stage}: selected model → {best_model.name}")
    logger.info(f"{stage}: SUCCESS")