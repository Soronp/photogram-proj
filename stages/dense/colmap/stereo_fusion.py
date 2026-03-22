from pathlib import Path
import open3d as o3d


def run(paths, config, logger, tool_runner):
    stage = "stereo_fusion"
    logger.info(f"---- {stage.upper()} ----")

    dense_dir = paths.dense
    output_path = dense_dir / "fused.ply"

    if not dense_dir.exists():
        raise RuntimeError(f"{stage}: dense workspace missing")

    retry = config.get("_meta", {}).get("retry_count", 0)
    use_gpu = config.get("dense", {}).get("use_gpu", True)

    # =====================================================
    # EDGE-PRESERVING PARAMETERS
    # =====================================================
    min_pixels = 2 if retry == 0 else max(1, 2 - retry)
    max_reproj = 2.0 + 0.5 * retry
    max_depth = 0.01 + 0.01 * retry
    max_normal = 10 + retry * 5

    logger.info(
        f"{stage}: min_pixels={min_pixels}, max_reproj={max_reproj}, "
        f"max_depth={max_depth}, max_normal={max_normal}"
    )

    # =====================================================
    # BUILD CMD
    # =====================================================
    def _build_cmd(use_gpu_flag=True):
        gpu_idx = "0" if use_gpu_flag else "-1"

        return [
            "colmap",
            "stereo_fusion",
            "--workspace_path", str(dense_dir),
            "--workspace_format", "COLMAP",
            "--input_type", "geometric",
            "--output_type", "PLY",
            "--output_path", str(output_path),
            "--StereoFusion.num_threads", "-1",
            "--StereoFusion.cache_size", "256",
            "--StereoFusion.min_num_pixels", str(min_pixels),
            "--StereoFusion.max_reproj_error", str(max_reproj),
            "--StereoFusion.max_depth_error", str(max_depth),
            "--StereoFusion.max_normal_error", str(max_normal),
        ]

    # =====================================================
    # EXECUTION
    # =====================================================
    try:
        if use_gpu:
            logger.info(f"{stage}: GPU fusion...")
            tool_runner.run(_build_cmd(True), stage=stage + "_gpu")
        else:
            raise RuntimeError("GPU disabled")
    except Exception as e:
        logger.warning(f"{stage}: GPU failed → {e}")
        logger.info(f"{stage}: CPU fallback...")
        tool_runner.run(_build_cmd(False), stage=stage + "_cpu")

    # =====================================================
    # VALIDATION
    # =====================================================
    if not output_path.exists():
        raise RuntimeError(f"{stage}: fusion failed → no output")

    pcd = o3d.io.read_point_cloud(str(output_path))
    num_points = len(pcd.points)

    logger.info(f"{stage}: fused points = {num_points}")

    if num_points < 500_000:
        status = "low"
        logger.warning(f"{stage}: low density → missing details likely")
    elif num_points < 1_000_000:
        status = "moderate"
        logger.info(f"{stage}: moderate density")
    elif num_points < 2_500_000:
        status = "good"
        logger.info(f"{stage}: good density")
    else:
        status = "very_high"
        logger.info(f"{stage}: very dense → excellent detail")

    logger.info(f"{stage}: SUCCESS")
    return {"num_points": int(num_points), "status": status}