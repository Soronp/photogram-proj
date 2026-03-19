from pathlib import Path


def run(paths, config, logger, tool_runner):
    stage = "stereo_fusion"
    logger.info(f"---- {stage.upper()} ----")

    dense_dir = paths.dense

    if not dense_dir.exists():
        raise RuntimeError(f"{stage}: dense dir missing")

    output_path = dense_dir / "fused.ply"

    if output_path.exists():
        logger.warning(f"{stage}: already exists, skipping")
        return

    use_gpu = config.get("dense", {}).get("use_gpu", True)

    # -----------------------------
    # COMMAND (BALANCED DENSITY)
    # -----------------------------
    cmd = [
        "colmap",
        "stereo_fusion",

        "--workspace_path", str(dense_dir),
        "--workspace_format", "COLMAP",

        "--input_type", "geometric",
        "--output_type", "PLY",

        "--output_path", str(output_path),

        # 🔥 KEEP MORE POINTS (BUT NOT NOISY)
        "--StereoFusion.min_num_pixels", "3",

        "--StereoFusion.max_reproj_error", "2.5",
        "--StereoFusion.max_depth_error", "0.02",
        "--StereoFusion.max_normal_error", "10",

        # 🔥 PERFORMANCE
        "--StereoFusion.num_threads", "-1",
        "--StereoFusion.max_image_size", "2000",
    ]

    tool_runner.run(cmd, stage=stage)

    if not output_path.exists():
        raise RuntimeError(f"{stage}: fusion failed")

    # -----------------------------
    # DEBUG METRIC
    # -----------------------------
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(str(output_path))

    logger.info(f"{stage}: fused points = {len(pcd.points)}")

    if len(pcd.points) < 300000:
        logger.warning(f"{stage}: LOW DENSITY POINT CLOUD")

    logger.info(f"{stage}: SUCCESS")