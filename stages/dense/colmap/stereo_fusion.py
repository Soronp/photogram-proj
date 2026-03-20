from pathlib import Path
import open3d as o3d


def run(paths, config, logger, tool_runner):
    stage = "stereo_fusion"
    logger.info(f"---- {stage.upper()} ----")

    dense_dir = paths.dense

    if not dense_dir.exists():
        raise RuntimeError(f"{stage}: dense dir missing")

    output_path = dense_dir / "fused.ply"

    # 🔥 RETRY-AWARE CLEANUP
    retry_count = config.get("_meta", {}).get("retry_count", 0)

    if output_path.exists():
        logger.warning(f"{stage}: removing previous fusion (retry={retry_count})")
        output_path.unlink()

    # -----------------------------
    # 🔥 ANALYSIS SIGNALS (CORRECT STRUCTURE)
    # -----------------------------
    analysis = config.get("analysis_results", {})

    matches = analysis.get("matches", {})

    connectivity = matches.get("connectivity", 0.3)
    avg_degree = matches.get("avg_degree", 4)

    logger.info(f"{stage}: connectivity={connectivity}, avg_degree={avg_degree}")

    # -----------------------------
    # 🔥 FUSION PARAMETERS
    # -----------------------------
    fusion_cfg = config.get("fusion", {})

    min_pixels = fusion_cfg.get("min_num_pixels", 5)

    if connectivity < 0.2:
        max_reproj = 3.0
        max_depth = 0.05
        mode = "weak"

    elif connectivity < 0.4:
        max_reproj = 2.5
        max_depth = 0.03
        mode = "moderate"

    else:
        max_reproj = 2.0
        max_depth = 0.02
        mode = "strong"

    logger.info(f"{stage}: mode={mode}")

    # -----------------------------
    # 🔥 COMMAND
    # -----------------------------
    cmd = [
        "colmap",
        "stereo_fusion",

        "--workspace_path", str(dense_dir),
        "--workspace_format", "COLMAP",

        "--input_type", "geometric",
        "--output_type", "PLY",

        "--output_path", str(output_path),

        "--StereoFusion.min_num_pixels", str(min_pixels),
        "--StereoFusion.max_reproj_error", str(max_reproj),
        "--StereoFusion.max_depth_error", str(max_depth),
        "--StereoFusion.max_normal_error", "10",

        "--StereoFusion.num_threads", "-1",
    ]

    tool_runner.run(cmd, stage=stage)

    if not output_path.exists():
        raise RuntimeError(f"{stage}: fusion failed")

    # -----------------------------
    # 🔥 METRICS
    # -----------------------------
    pcd = o3d.io.read_point_cloud(str(output_path))
    num_points = len(pcd.points)

    logger.info(f"{stage}: fused points = {num_points}")

    # Better thresholds
    if num_points < 100000:
        logger.warning(f"{stage}: VERY LOW density")
    elif num_points < 300000:
        logger.warning(f"{stage}: moderate density")
    else:
        logger.info(f"{stage}: GOOD density")

    logger.info(f"{stage}: SUCCESS")