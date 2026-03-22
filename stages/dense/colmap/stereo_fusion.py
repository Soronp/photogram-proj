from pathlib import Path
import open3d as o3d


def run(paths, config, logger, tool_runner):
    stage = "stereo_fusion"
    logger.info(f"---- {stage.upper()} ----")

    dense_dir = paths.dense
    output_path = dense_dir / "fused.ply"

    if not dense_dir.exists():
        raise RuntimeError(f"{stage}: dense workspace missing")

    # =====================================================
    # 🔥 ANALYSIS INPUT
    # =====================================================
    analysis = config.get("analysis_results", {})
    matches = analysis.get("matches", {})

    connectivity = matches.get("connectivity", 0.3)
    avg_degree = matches.get("avg_degree", 4)
    retry = config.get("_meta", {}).get("retry_count", 0)

    logger.info(
        f"{stage}: connectivity={connectivity:.3f}, "
        f"avg_degree={avg_degree}, retry={retry}"
    )

    # =====================================================
    # 🔥 CORE STRATEGY: DENSITY-FIRST FUSION
    # =====================================================

    # MUCH more permissive base
    if connectivity < 0.2:
        max_reproj = 3.5
        max_depth = 0.08
        min_pixels = 1
        max_normal = 25
        mode = "weak_geometry_dense"

    elif connectivity < 0.4:
        max_reproj = 3.0
        max_depth = 0.06
        min_pixels = 1
        max_normal = 20
        mode = "moderate_dense"

    else:
        max_reproj = 2.6
        max_depth = 0.05
        min_pixels = 2
        max_normal = 18
        mode = "strong_dense"

    # =====================================================
    # 🔥 RETRY → CONTROLLED DENSITY EXPANSION
    # =====================================================

    if retry > 0:
        logger.info(f"{stage}: retry expansion active")

        max_reproj = min(4.0, max_reproj + 0.3 * retry)
        max_depth = min(0.10, max_depth + 0.015 * retry)
        min_pixels = max(1, min_pixels - retry)
        max_normal = min(30, max_normal + 2 * retry)

    logger.info(
        f"{stage}: mode={mode}, "
        f"min_pixels={min_pixels}, "
        f"max_reproj={max_reproj:.3f}, "
        f"max_depth={max_depth:.3f}, "
        f"max_normal={max_normal}"
    )

    # =====================================================
    # 🔥 GPU CONFIG
    # =====================================================

    use_gpu = config.get("dense", {}).get("use_gpu", True)

    def _build_cmd(use_gpu=True):
        gpu_idx = "0" if use_gpu else "-1"

        return [
            "colmap",
            "stereo_fusion",
            "--workspace_path", str(dense_dir),
            "--workspace_format", "COLMAP",
            "--input_type", "geometric",
            "--output_type", "PLY",
            "--output_path", str(output_path),

            # 🔥 DENSITY CONTROLS (CRITICAL)
            "--StereoFusion.min_num_pixels", str(min_pixels),
            "--StereoFusion.max_reproj_error", str(max_reproj),
            "--StereoFusion.max_depth_error", str(max_depth),
            "--StereoFusion.max_normal_error", str(max_normal),

            # 🔥 PERFORMANCE + STABILITY
            "--StereoFusion.cache_size", "128",
            "--StereoFusion.num_threads", "-1",
        ]

    # =====================================================
    # 🔥 EXECUTION
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
    # 🔥 VALIDATION
    # =====================================================

    if not output_path.exists():
        raise RuntimeError(f"{stage}: fusion failed → no output")

    pcd = o3d.io.read_point_cloud(str(output_path))
    num_points = len(pcd.points)

    logger.info(f"{stage}: fused points = {num_points}")

    # =====================================================
    # 🔥 TARGET-DRIVEN CLASSIFICATION
    # =====================================================

    if num_points < 800_000:
        status = "too_low"
        logger.warning(f"{stage}: below target → needs more density")

    elif num_points > 3_000_000:
        status = "too_high"
        logger.warning(f"{stage}: too dense → may slow meshing")

    else:
        status = "good"
        logger.info(f"{stage}: ✅ TARGET ACHIEVED (1M–3M range)")

    logger.info(f"{stage}: SUCCESS")

    return {
        "num_points": int(num_points),
        "status": status,
        "connectivity": float(connectivity),
        "avg_degree": float(avg_degree),
    }