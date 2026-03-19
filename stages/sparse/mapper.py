from pathlib import Path

def run(paths, config, logger, tool_runner):
    stage = "mapper"
    logger.info(f"---- {stage.upper()} ----")

    sparse_root = paths.sparse
    image_dir = paths.images
    database_path = paths.database

    if not image_dir.exists():
        raise RuntimeError(f"{stage}: image directory not found")

    if not database_path.exists():
        raise RuntimeError(f"{stage}: database missing")

    sparse_root.mkdir(parents=True, exist_ok=True)

    logger.info(f"{stage}: running COLMAP mapper")

    # 🔥 VERY RELAXED MAPPER SETTINGS
    cmd = [
        "colmap",
        "mapper",
        "--database_path", str(database_path),
        "--image_path", str(image_dir),
        "--output_path", str(sparse_root),

        "--Mapper.num_threads", "-1",

        # 🔥 CRITICAL (LOWER THESE)
        "--Mapper.init_min_num_inliers", "15",
        "--Mapper.abs_pose_min_num_inliers", "15",

        # 🔥 ALLOW HARD DATASETS
        "--Mapper.ba_global_max_num_iterations", "100",
        "--Mapper.ba_local_max_num_iterations", "50",

        # 🔥 KEEP MORE IMAGES
        "--Mapper.min_model_size", "5",

        # 🔥 IMPORTANT
        "--Mapper.multiple_models", "0",
    ]

    tool_runner.run(cmd, stage=stage)

    # -----------------------------
    # Validate output
    # -----------------------------
    sparse_model = paths.sparse / "0"

    if not sparse_model.exists():
        raise RuntimeError(f"{stage}: no sparse model folder created")

    files = list(sparse_model.iterdir())
    if len(files) == 0:
        raise RuntimeError(f"{stage}: sparse model is empty")

    logger.info(f"{stage}: SUCCESS — sparse model created")