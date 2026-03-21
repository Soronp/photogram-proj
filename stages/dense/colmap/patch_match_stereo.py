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
    # 🔥 ANALYSIS INPUT
    # =====================================================
    analysis = config.get("analysis_results", {})
    matches = analysis.get("matches", {})
    features = analysis.get("features", {})

    connectivity = matches.get("connectivity", 0.3)
    avg_degree = matches.get("avg_degree", 4)
    feature_density = features.get("feature_density", 0.003)
    retry = config.get("_meta", {}).get("retry_count", 0)

    # =====================================================
    # 🔥 TARGET: ~1M POINTS (CONTROLLED)
    # =====================================================
    # Keep PatchMatch MODERATE → control density in fusion
    if retry == 0:
        window_radius = 6
        num_samples = 15
        num_iterations = 5
        logger.info(f"{stage}: FAST BALANCED MODE (~1M target)")

    elif retry == 1:
        window_radius = 7
        num_samples = 18
        num_iterations = 6
        logger.info(f"{stage}: QUALITY BOOST MODE")

    else:
        window_radius = 8
        num_samples = 20
        num_iterations = 6
        logger.info(f"{stage}: FINAL ATTEMPT MODE")

    # =====================================================
    # 🔥 DATA-DRIVEN LIGHT ADAPTATION (NOT EXPLOSIVE)
    # =====================================================
    if avg_degree < 3:
        window_radius += 1
        num_samples += 3
        logger.info(f"{stage}: weak geometry → slight boost")

    if feature_density < 0.001:
        window_radius += 1
        logger.info(f"{stage}: low texture → slightly larger window")

    # Hard caps (CRITICAL)
    window_radius = min(window_radius, 9)
    num_samples = min(num_samples, 24)

    filter_consistent = 2 if connectivity < 0.25 else 3

    # =====================================================
    # 🔥 GPU CONTROL
    # =====================================================
    use_gpu = config.get("dense", {}).get("use_gpu", True)

    def _build_cmd(use_gpu=True):
        gpu_idx = "0" if use_gpu else "-1"

        return [
            "colmap",
            "patch_match_stereo",
            "--workspace_path", str(dense_dir),
            "--workspace_format", "COLMAP",

            "--PatchMatchStereo.geom_consistency", "1",

            # 🔥 KEY PERFORMANCE CONTROL
            "--PatchMatchStereo.max_image_size", "2000",

            "--PatchMatchStereo.window_radius", str(window_radius),
            "--PatchMatchStereo.num_samples", str(num_samples),
            "--PatchMatchStereo.num_iterations", str(num_iterations),

            "--PatchMatchStereo.filter", "1",
            "--PatchMatchStereo.filter_min_num_consistent", str(filter_consistent),

            "--PatchMatchStereo.gpu_index", gpu_idx,

            # 🔥 MEMORY + SPEED
            "--PatchMatchStereo.cache_size", "64",
        ]

    # =====================================================
    # 🔥 EXECUTION
    # =====================================================
    try:
        if use_gpu:
            logger.info(f"{stage}: GPU execution...")
            tool_runner.run(_build_cmd(True), stage=stage + "_gpu")
        else:
            raise RuntimeError("GPU disabled")

    except Exception as e:
        logger.warning(f"{stage}: GPU failed → CPU fallback ({e})")
        tool_runner.run(_build_cmd(False), stage=stage + "_cpu")

    # =====================================================
    # 🔥 VALIDATION
    # =====================================================
    depth_dir = dense_dir / "stereo" / "depth_maps"

    if not depth_dir.exists():
        raise RuntimeError(f"{stage}: depth_maps missing")

    depth_maps = list(depth_dir.glob("*.bin"))
    num_images = len(list(images_dir.glob("*")))
    coverage = len(depth_maps) / max(num_images, 1)

    logger.info(f"{stage}: coverage = {coverage:.2f}")

    if coverage < 0.4:
        logger.warning(f"{stage}: LOW coverage")
    else:
        logger.info(f"{stage}: OK coverage")

    logger.info(f"{stage}: SUCCESS")