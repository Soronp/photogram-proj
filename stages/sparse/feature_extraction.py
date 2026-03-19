from pathlib import Path

def run(paths, config, logger, tool_runner):
    stage = "feature_extraction"
    logger.info(f"---- {stage.upper()} ----")

    # 🔥 ALWAYS use canonical image folder
    image_dir = paths.images

    if not image_dir.exists():
        raise RuntimeError(f"{stage}: image directory not found: {image_dir}")

    images = list(image_dir.iterdir())
    if len(images) == 0:
        raise RuntimeError(f"{stage}: no images found in {image_dir}")

    logger.info(f"{stage}: using {len(images)} images")

    # -----------------------------
    # Database
    # -----------------------------
    database_path = paths.database

    if database_path.exists():
        logger.warning(f"{stage}: database exists, deleting for clean run")
        database_path.unlink()

    # -----------------------------
    # SIFT config (MAXIMIZE FEATURES)
    # -----------------------------
    sift_config = config.get("sift", {})

    cmd = [
        "colmap",
        "feature_extractor",
        "--database_path", str(database_path),
        "--image_path", str(image_dir),

        # 🔥 MAX ROBUSTNESS SETTINGS
        "--SiftExtraction.use_gpu", "0",
        "--SiftExtraction.max_num_features", str(sift_config.get("max_num_features", 12000)),
        "--SiftExtraction.max_image_size", str(sift_config.get("max_image_size", 2000)),
        "--SiftExtraction.num_threads", "-1",

        # 🔥 IMPORTANT
        "--SiftExtraction.estimate_affine_shape", "1",
        "--SiftExtraction.domain_size_pooling", "1",
    ]

    tool_runner.run(cmd, stage=stage)

    logger.info(f"{stage}: DONE")