from pathlib import Path


def run(paths, config, logger, tool_runner):
    stage = "matching"
    logger.info(f"---- {stage.upper()} ----")

    database_path = paths.database

    if not database_path.exists():
        raise RuntimeError(f"{stage}: database.db not found. Run feature extraction first.")

    # -----------------------------
    # Config
    # -----------------------------
    match_config = config.get("matching", {})

    matcher_type = match_config.get("type", "exhaustive")
    use_gpu = match_config.get("use_gpu", False)

    gpu_flag = 1 if use_gpu else 0

    # -----------------------------
    # Select matcher
    # -----------------------------
    if matcher_type == "exhaustive":
        cmd = [
            "colmap",
            "exhaustive_matcher",
            "--database_path", str(database_path),

            "--SiftMatching.use_gpu", str(gpu_flag),
            "--SiftMatching.num_threads", "-1",
            "--SiftMatching.max_ratio", "0.8",
            "--SiftMatching.max_distance", "0.7",
            "--SiftMatching.cross_check", "1",
        ]

    else:
        raise ValueError(f"{stage}: Unsupported matcher type: {matcher_type}")

    # -----------------------------
    # Run
    # -----------------------------
    tool_runner.run(cmd, stage=stage)

    logger.info(f"{stage}: completed successfully")