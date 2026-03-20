def run(paths, config, logger, tool_runner):
    stage = "feature_matching"
    logger.info(f"---- {stage.upper()} ----")

    database_path = paths.database

    if not database_path.exists():
        raise RuntimeError(f"{stage}: database missing")

    matching = config.get("matching", {})
    matcher_type = matching.get("type", "exhaustive")
    backend = config.get("sparse", {}).get("backend", "colmap")

    logger.info(f"{stage}: backend = {backend}, matcher = {matcher_type}")

    # =====================================================
    # 🔥 GLOMAP → STRICT MATCHING ONLY
    # =====================================================
    if backend == "glomap":
        logger.info(f"{stage}: enforcing strict matching for GLOMAP")

        cmd = [
            "colmap", "exhaustive_matcher",

            "--database_path", str(database_path),

            "--SiftMatching.use_gpu", "0",
            "--SiftMatching.num_threads", "-1",

            "--SiftMatching.max_ratio", "0.75",
            "--SiftMatching.max_distance", "0.7",
            "--SiftMatching.cross_check", "1",

            "--SiftMatching.guided_matching", "1",
        ]

    # =====================================================
    # 🔷 COLMAP (adaptive)
    # =====================================================
    else:
        if matcher_type == "exhaustive":
            cmd = [
                "colmap", "exhaustive_matcher",

                "--database_path", str(database_path),

                "--SiftMatching.use_gpu", "0",
                "--SiftMatching.num_threads", "-1",

                "--SiftMatching.max_ratio", "0.85",
                "--SiftMatching.max_distance", "0.8",
                "--SiftMatching.cross_check", "1",

                "--SiftMatching.guided_matching", "1",
            ]

        elif matcher_type == "sequential":
            cmd = [
                "colmap", "sequential_matcher",
                "--database_path", str(database_path),
                "--SiftMatching.guided_matching", "1",
            ]

        elif matcher_type == "vocab_tree":
            cmd = [
                "colmap", "vocab_tree_matcher",
                "--database_path", str(database_path),
            ]

        else:
            raise ValueError(f"{stage}: unknown matcher type")

    tool_runner.run(cmd, stage=stage)

    logger.info(f"{stage}: DONE")