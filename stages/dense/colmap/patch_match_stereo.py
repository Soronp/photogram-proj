from pathlib import Path


def run(paths, config, logger, tool_runner):
    stage = "patch_match_stereo"
    logger.info(f"---- {stage.upper()} ----")

    dense_dir = paths.dense

    if not dense_dir.exists():
        raise RuntimeError(f"{stage}: dense workspace missing")

    images_dir = dense_dir / "images"
    sparse_dir = dense_dir / "sparse"

    if not images_dir.exists() or not any(images_dir.iterdir()):
        raise RuntimeError(f"{stage}: undistorted images missing")

    if not sparse_dir.exists():
        raise RuntimeError(f"{stage}: dense sparse model missing")

    # -----------------------------
    # CONFIG (BALANCED FOR 2K IMAGES)
    # -----------------------------
    use_gpu = config.get("dense", {}).get("use_gpu", True)
    gpu_index = "0" if use_gpu else "-1"

    logger.info(f"{stage}: GPU enabled = {use_gpu}")

    # -----------------------------
    # COMMAND (TUNED)
    # -----------------------------
    cmd = [
        "colmap",
        "patch_match_stereo",

        "--workspace_path", str(dense_dir),
        "--workspace_format", "COLMAP",

        # 🔥 CORE SETTINGS
        "--PatchMatchStereo.geom_consistency", "1",
        "--PatchMatchStereo.max_image_size", "2000",

        # 🔥 BALANCED MATCHING (IMPORTANT)
        "--PatchMatchStereo.window_radius", "5",
        "--PatchMatchStereo.num_samples", "15",
        "--PatchMatchStereo.num_iterations", "5",

        # 🔥 FILTERING (CRITICAL)
        "--PatchMatchStereo.filter", "1",
        "--PatchMatchStereo.filter_min_num_consistent", "2",

        # 🔥 GPU
        "--PatchMatchStereo.gpu_index", gpu_index,

        # 🔥 PERFORMANCE
        "--PatchMatchStereo.cache_size", "32",
    ]

    tool_runner.run(cmd, stage=stage)

    # -----------------------------
    # VALIDATE OUTPUT
    # -----------------------------
    depth_dir = dense_dir / "stereo" / "depth_maps"

    if not depth_dir.exists():
        raise RuntimeError(f"{stage}: depth_maps folder missing")

    depth_maps = list(depth_dir.glob("*.bin"))

    num_images = len(list(images_dir.glob("*")))
    num_depths = len(depth_maps)

    logger.info(f"{stage}: images = {num_images}")
    logger.info(f"{stage}: depth maps = {num_depths}")

    if num_depths < num_images * 0.7:
        logger.warning(f"{stage}: LOW depth coverage ({num_depths}/{num_images})")

    logger.info(f"{stage}: SUCCESS")