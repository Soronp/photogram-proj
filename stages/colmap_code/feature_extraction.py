from pathlib import Path


def run(paths, config, logger, tool_runner):
    stage = "feature_extraction"

    logger.info(f"---- {stage.upper()} ----")

    # -----------------------------
    # Select image directory
    # -----------------------------
    downsample_enabled = config.get("downsampling", {}).get("enabled", True)

    if downsample_enabled:
        image_dir = paths.images_downsampled
    else:
        image_dir = paths.images

    if not image_dir.exists():
        raise RuntimeError(f"Image directory not found: {image_dir}")

    # -----------------------------
    # Database path
    # -----------------------------
    database_path = paths.database

    # Optional: remove old database if force=True later
    if database_path.exists():
        logger.warning(f"{stage}: database already exists, skipping...")
        return

    # -----------------------------
    # Config parameters
    # -----------------------------
    sift_config = config.get("sift", {})

    max_features = sift_config.get("max_num_features", 8192)
    max_image_size = sift_config.get("max_image_size", 3200)
    num_threads = sift_config.get("num_threads", -1)

    use_gpu = sift_config.get("use_gpu", False)
    gpu_flag = 1 if use_gpu else 0

    # -----------------------------
    # Build command
    # -----------------------------
    cmd = [
        "colmap",
        "feature_extractor",
        "--database_path", str(database_path),
        "--image_path", str(image_dir),

        "--SiftExtraction.use_gpu", str(gpu_flag),
        "--SiftExtraction.max_num_features", str(max_features),
        "--SiftExtraction.max_image_size", str(max_image_size),
        "--SiftExtraction.num_threads", str(num_threads),
    ]

    # -----------------------------
    # Run
    # -----------------------------
    tool_runner.run(cmd, stage=stage)

    logger.info(f"{stage}: completed successfully")