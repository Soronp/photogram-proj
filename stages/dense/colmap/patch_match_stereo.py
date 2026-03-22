from pathlib import Path


def run(paths, config, logger, tool_runner):
    stage = "patch_match_stereo"
    logger.info(f"---- {stage.upper()} ----")

    dense_dir = paths.dense
    images_dir = dense_dir / "images"
    sparse_dir = dense_dir / "sparse"

    if not images_dir.exists() or not any(images_dir.iterdir()):
        raise RuntimeError(f"{stage}: undistorted images missing")
    if not sparse_dir.exists():
        raise RuntimeError(f"{stage}: sparse model missing")

    # =====================================================
    # META / CONFIG
    # =====================================================
    retry = config.get("_meta", {}).get("retry_count", 0)
    use_gpu = config.get("dense", {}).get("use_gpu", True)

    # Base parameters
    window_radius = 5 + retry
    num_samples = 15 + retry * 2
    num_iterations = 5 + retry
    max_img_size = 2400 + retry * 400
    geom_consistency = 1
    filter_enabled = 1
    filter_min_num_consistent = 2 + retry

    logger.info(
        f"{stage}: radius={window_radius}, samples={num_samples}, "
        f"iterations={num_iterations}, max_size={max_img_size}, retry={retry}"
    )

    # =====================================================
    # BUILD CMD
    # =====================================================
    def _build_cmd(use_gpu_flag=True):
        gpu_idx = "0" if use_gpu_flag else "-1"

        return [
            "colmap",
            "patch_match_stereo",
            "--workspace_path", str(dense_dir),
            "--workspace_format", "COLMAP",
            "--PatchMatchStereo.max_image_size", str(max_img_size),
            "--PatchMatchStereo.window_radius", str(window_radius),
            "--PatchMatchStereo.num_samples", str(num_samples),
            "--PatchMatchStereo.num_iterations", str(num_iterations),
            "--PatchMatchStereo.geom_consistency", str(geom_consistency),
            "--PatchMatchStereo.filter", str(filter_enabled),
            "--PatchMatchStereo.filter_min_num_consistent", str(filter_min_num_consistent),
            "--PatchMatchStereo.gpu_index", gpu_idx,
        ]

    # =====================================================
    # EXECUTION
    # =====================================================
    try:
        if use_gpu:
            logger.info(f"{stage}: running GPU PatchMatch...")
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
    depth_dir = dense_dir / "stereo" / "depth_maps"
    if not depth_dir.exists():
        raise RuntimeError(f"{stage}: depth maps missing")

    depth_maps = list(depth_dir.glob("*.bin"))
    coverage = len(depth_maps) / max(len(list(images_dir.glob("*"))), 1)

    logger.info(f"{stage}: coverage = {coverage:.2f}")
    if coverage < 0.6:
        logger.warning(f"{stage}: LOW coverage → missing geometry likely")
    elif coverage < 0.85:
        logger.info(f"{stage}: moderate coverage")
    else:
        logger.info(f"{stage}: excellent coverage")

    logger.info(f"{stage}: SUCCESS")