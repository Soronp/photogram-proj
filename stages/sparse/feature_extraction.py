from pathlib import Path


def run(paths, config, logger, tool_runner):
    stage = "feature_extraction"
    logger.info(f"---- {stage.upper()} ----")

    image_dir = paths.images
    database_path = paths.database

    if not image_dir.exists():
        raise RuntimeError(f"{stage}: image directory not found")

    images = list(image_dir.glob("*"))
    if not images:
        raise RuntimeError(f"{stage}: no images found")

    logger.info(f"{stage}: {len(images)} images")

    # Reset database
    if database_path.exists():
        logger.warning(f"{stage}: removing old database")
        database_path.unlink()

    sift = config.get("sift", {})
    backend = config.get("sparse", {}).get("backend", "colmap")

    # 🔥 Backend-aware tuning
    if backend == "glomap":
        peak_threshold = sift.get("peak_threshold", 0.006)  # stricter
        logger.info(f"{stage}: GLOMAP mode → stricter features")
    else:
        peak_threshold = sift.get("peak_threshold", 0.004)
        logger.info(f"{stage}: COLMAP mode → robust features")

    cmd = [
        "colmap", "feature_extractor",

        "--database_path", str(database_path),
        "--image_path", str(image_dir),

        "--SiftExtraction.use_gpu", str(int(sift.get("use_gpu", False))),
        "--SiftExtraction.max_num_features", str(sift.get("max_num_features", 10000)),
        "--SiftExtraction.max_image_size", str(sift.get("max_image_size", 2000)),
        "--SiftExtraction.num_threads", "-1",

        "--SiftExtraction.peak_threshold", str(peak_threshold),

        "--SiftExtraction.estimate_affine_shape", "1",
        "--SiftExtraction.domain_size_pooling", "1",
    ]

    tool_runner.run(cmd, stage=stage)

    logger.info(f"{stage}: DONE")