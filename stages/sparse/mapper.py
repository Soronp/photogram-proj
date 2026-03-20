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

    backend = config.get("sparse", {}).get("backend", "colmap")

    logger.info(f"{stage}: backend = {backend}")

    # =====================================================
    # 🔷 GLOMAP BACKEND
    # =====================================================
    if backend == "glomap":
        logger.info(f"{stage}: running GLOMAP mapper")

        cmd = [
            "glomap", "mapper",

            "--database_path", str(database_path),
            "--output_path", str(sparse_root),
        ]

        tool_runner.run(cmd, stage=stage + "_glomap")

    # =====================================================
    # 🔷 COLMAP BACKEND
    # =====================================================
    else:
        logger.info(f"{stage}: running COLMAP mapper")

        # 🔥 Read analysis signals
        analysis = config.get("analysis", {})
        connectivity = analysis.get("connectivity", 0.3)
        avg_degree = analysis.get("avg_degree", 4)

        # Adaptive tuning
        if connectivity < 0.2:
            init_inliers = 12
            abs_inliers = 12
            min_model_size = 3
            ba_global_iter = 150

        elif connectivity < 0.4:
            init_inliers = 20
            abs_inliers = 15
            min_model_size = 5
            ba_global_iter = 100

        else:
            init_inliers = 30
            abs_inliers = 20
            min_model_size = 10
            ba_global_iter = 50

        ba_local_iter = 50 if avg_degree < 4 else 30

        cmd = [
            "colmap", "mapper",

            "--database_path", str(database_path),
            "--image_path", str(image_dir),
            "--output_path", str(sparse_root),

            "--Mapper.num_threads", "-1",

            "--Mapper.init_min_num_inliers", str(init_inliers),
            "--Mapper.abs_pose_min_num_inliers", str(abs_inliers),

            "--Mapper.min_model_size", str(min_model_size),

            "--Mapper.ba_global_max_num_iterations", str(ba_global_iter),
            "--Mapper.ba_local_max_num_iterations", str(ba_local_iter),

            "--Mapper.multiple_models", "0",
        ]

        tool_runner.run(cmd, stage=stage + "_colmap")

    # =====================================================
    # VALIDATION (SHARED)
    # =====================================================
    sparse_model = sparse_root / "0"

    if not sparse_model.exists():
        raise RuntimeError(f"{stage}: no sparse model produced")

    files = list(sparse_model.glob("*"))
    if not files:
        raise RuntimeError(f"{stage}: empty sparse model")

    logger.info(f"{stage}: SUCCESS — sparse model ready")