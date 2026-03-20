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
        raise RuntimeError(f"{stage}: sparse model missing")

    # -----------------------------
    # 🔥 ANALYSIS SIGNALS
    # -----------------------------
    analysis = config.get("analysis", {})

    connectivity = analysis.get("connectivity", 0.3)
    avg_degree = analysis.get("avg_degree", 4)
    feature_density = analysis.get("feature_density", 0.003)

    logger.info(
        f"{stage}: connectivity={connectivity}, avg_degree={avg_degree}, "
        f"feature_density={feature_density}"
    )

    # -----------------------------
    # 🔥 ADAPTIVE PARAMS
    # -----------------------------
    if avg_degree < 3:
        # Weak geometry → explore more
        window_radius = 7
        num_samples = 25
        num_iterations = 7

    else:
        # Stable graph
        window_radius = 5
        num_samples = 15
        num_iterations = 5

    # Texture-based tuning
    if feature_density < 0.001:
        window_radius += 2  # larger patch helps low-texture
        logger.info(f"{stage}: low texture → increasing window size")

    # Filtering (critical)
    if connectivity < 0.2:
        filter_consistent = 2
    else:
        filter_consistent = 3

    # GPU
    use_gpu = config.get("dense", {}).get("use_gpu", True)
    gpu_index = "0" if use_gpu else "-1"

    logger.info(f"{stage}: GPU = {use_gpu}")

    # -----------------------------
    # 🔥 COMMAND
    # -----------------------------
    cmd = [
        "colmap",
        "patch_match_stereo",

        "--workspace_path", str(dense_dir),
        "--workspace_format", "COLMAP",

        "--PatchMatchStereo.geom_consistency", "1",
        "--PatchMatchStereo.max_image_size", str(
            config.get("dense", {}).get("max_image_size", 2000)
        ),

        "--PatchMatchStereo.window_radius", str(window_radius),
        "--PatchMatchStereo.num_samples", str(num_samples),
        "--PatchMatchStereo.num_iterations", str(num_iterations),

        "--PatchMatchStereo.filter", "1",
        "--PatchMatchStereo.filter_min_num_consistent", str(filter_consistent),

        "--PatchMatchStereo.gpu_index", gpu_index,
        "--PatchMatchStereo.cache_size", "32",
    ]

    tool_runner.run(cmd, stage=stage)

    # -----------------------------
    # VALIDATION
    # -----------------------------
    depth_dir = dense_dir / "stereo" / "depth_maps"

    if not depth_dir.exists():
        raise RuntimeError(f"{stage}: depth_maps missing")

    depth_maps = list(depth_dir.glob("*.bin"))
    num_images = len(list(images_dir.glob("*")))

    logger.info(f"{stage}: images = {num_images}")
    logger.info(f"{stage}: depth maps = {len(depth_maps)}")

    coverage = len(depth_maps) / max(num_images, 1)

    if coverage < 0.5:
        logger.warning(f"{stage}: VERY LOW depth coverage ({coverage:.2f})")
    elif coverage < 0.8:
        logger.warning(f"{stage}: moderate depth coverage ({coverage:.2f})")

    logger.info(f"{stage}: SUCCESS")