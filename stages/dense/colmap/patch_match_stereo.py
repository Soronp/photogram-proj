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

    logger.info(
        f"{stage}: connectivity={connectivity:.3f}, "
        f"avg_degree={avg_degree}, "
        f"feature_density={feature_density:.5f}, "
        f"retry={retry}"
    )

    # =====================================================
    # 🔥 CORE STRATEGY: CONTROLLED PATCHMATCH
    # =====================================================
    # DO NOT SCALE EXPONENTIALLY → keep runtime stable

    if retry == 0:
        window_radius = 6
        num_samples = 12
        num_iterations = 4
        max_img_size = 1600
        logger.info(f"{stage}: FAST BASE MODE")

    elif retry == 1:
        window_radius = 7
        num_samples = 16
        num_iterations = 5
        max_img_size = 1800
        logger.info(f"{stage}: BALANCED MODE")

    else:
        window_radius = 8
        num_samples = 20
        num_iterations = 6
        max_img_size = 2000
        logger.info(f"{stage}: HIGH QUALITY MODE")

    # =====================================================
    # 🔥 ADAPTIVE TUNING (SAFE ONLY)
    # =====================================================

    if avg_degree < 3:
        num_samples += 2
        logger.info(f"{stage}: weak geometry → slight sample boost")

    if feature_density < 0.001:
        window_radius += 1
        logger.info(f"{stage}: low texture → slightly larger window")

    # HARD LIMITS (prevent explosion)
    window_radius = min(window_radius, 9)
    num_samples = min(num_samples, 24)
    max_img_size = min(max_img_size, 2000)

    # =====================================================
    # 🔥 HOLE PREVENTION SETTINGS
    # =====================================================

    # IMPORTANT: this fixes your mesh holes
    geom_consistency = 1
    filter_enabled = 1

    # Less aggressive filtering = fewer holes
    filter_consistent = 2 if connectivity < 0.3 else 3

    # =====================================================
    # 🔥 GPU CONFIG
    # =====================================================

    use_gpu = config.get("dense", {}).get("use_gpu", True)

    def _build_cmd(use_gpu=True):
        gpu_idx = "0" if use_gpu else "-1"

        return [
            "colmap",
            "patch_match_stereo",
            "--workspace_path", str(dense_dir),
            "--workspace_format", "COLMAP",

            # 🔥 CRITICAL: CONTROL RUNTIME
            "--PatchMatchStereo.max_image_size", str(max_img_size),

            # 🔥 CORE PARAMS
            "--PatchMatchStereo.window_radius", str(window_radius),
            "--PatchMatchStereo.num_samples", str(num_samples),
            "--PatchMatchStereo.num_iterations", str(num_iterations),

            # 🔥 KEEP THIS ON (prevents garbage depth)
            "--PatchMatchStereo.geom_consistency", str(geom_consistency),

            # 🔥 HOLE CONTROL (LESS STRICT)
            "--PatchMatchStereo.filter", str(filter_enabled),
            "--PatchMatchStereo.filter_min_num_consistent", str(filter_consistent),

            # 🔥 STABILITY + SPEED
            "--PatchMatchStereo.cache_size", "64",

            "--PatchMatchStereo.gpu_index", gpu_idx,
        ]

    # =====================================================
    # 🔥 EXECUTION
    # =====================================================

    try:
        if use_gpu:
            logger.info(f"{stage}: running GPU PatchMatch...")
            tool_runner.run(_build_cmd(True), stage=stage + "_gpu")
        else:
            raise RuntimeError("GPU disabled")

    except Exception as e:
        logger.warning(f"{stage}: GPU failed → {e}")
        logger.info(f"{stage}: falling back to CPU...")
        tool_runner.run(_build_cmd(False), stage=stage + "_cpu")

    # =====================================================
    # 🔥 VALIDATION
    # =====================================================

    depth_dir = dense_dir / "stereo" / "depth_maps"

    if not depth_dir.exists():
        raise RuntimeError(f"{stage}: depth maps missing")

    depth_maps = list(depth_dir.glob("*.bin"))
    num_images = len(list(images_dir.glob("*")))
    coverage = len(depth_maps) / max(num_images, 1)

    logger.info(f"{stage}: coverage = {coverage:.2f}")

    if coverage < 0.5:
        logger.warning(f"{stage}: LOW coverage → expect holes")
    elif coverage < 0.75:
        logger.info(f"{stage}: moderate coverage")
    else:
        logger.info(f"{stage}: excellent coverage")

    logger.info(f"{stage}: SUCCESS")