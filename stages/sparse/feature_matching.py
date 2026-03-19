from pathlib import Path

def run(paths, config, logger, tool_runner):
    stage = "feature_matching"
    logger.info(f"---- {stage.upper()} ----")

    database_path = paths.database

    if not database_path.exists():
        raise RuntimeError(f"{stage}: database not found")

    logger.info(f"{stage}: running EXHAUSTIVE matching")

    # 🔥 VERY RELAXED matching for small dataset
    cmd = [
        "colmap",
        "exhaustive_matcher",
        "--database_path", str(database_path),

        "--SiftMatching.use_gpu", "0",
        "--SiftMatching.num_threads", "-1",

        # 🔥 RELAXED (CRITICAL)
        "--SiftMatching.max_ratio", "0.9",
        "--SiftMatching.max_distance", "0.9",
        "--SiftMatching.cross_check", "0",

        # 🔥 MORE MATCHES
        "--SiftMatching.guided_matching", "1",
    ]

    tool_runner.run(cmd, stage=stage)

    logger.info(f"{stage}: DONE")