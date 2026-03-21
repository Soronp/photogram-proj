from pathlib import Path
import open3d as o3d

def run(paths, config, logger, tool_runner):
    stage = "stereo_fusion"
    logger.info(f"---- {stage.upper()} ----")

    dense_dir = paths.dense
    if not dense_dir.exists():
        raise RuntimeError(f"{stage}: dense workspace missing")

    output_path = dense_dir / "fused.ply"

    # =====================================================
    # 🔥 ANALYSIS SIGNALS
    # =====================================================
    analysis = config.get("analysis_results", {})
    matches = analysis.get("matches", {})

    connectivity = matches.get("connectivity", 0.3)
    avg_degree = matches.get("avg_degree", 4)

    logger.info(f"{stage}: connectivity={connectivity}, avg_degree={avg_degree}")

    # =====================================================
    # 🔥 FUSION PARAMETERS (HIGH-FIRST STRATEGY → 1M points)
    # =====================================================
    fusion_cfg = config.get("fusion", {})
    min_pixels = fusion_cfg.get("min_num_pixels", 2)
    retry = config.get("_meta", {}).get("retry_count", 0)

    # Adaptive thresholds tuned for ~1M points
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

    # Retry adaptation: relax thresholds
    if retry > 0:
        relax_factor = 1 + 0.1 * retry
        max_reproj *= relax_factor
        max_depth *= relax_factor
        min_pixels = max(1, int(min_pixels * (1 - 0.05 * retry)))
        logger.info(f"{stage}: retry={retry} → thresholds relaxed")

    logger.info(f"{stage}: mode={mode}, min_pixels={min_pixels}, max_reproj={max_reproj}, max_depth={max_depth}")

    # =====================================================
    # 🔥 GPU CONFIG (PRIORITIZE GPU, FALLBACK CPU)
    # =====================================================
    use_gpu = config.get("dense", {}).get("use_gpu", True)
    logger.info(f"{stage}: initial GPU preference = {use_gpu}")

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
            "--StereoFusion.min_num_pixels", str(min_pixels),
            "--StereoFusion.max_reproj_error", str(max_reproj),
            "--StereoFusion.max_depth_error", str(max_depth),
            "--StereoFusion.max_normal_error", "10",
            "--StereoFusion.num_threads", "-1",
            "--StereoFusion.cache_size", "128",
        ]

    # =====================================================
    # 🔥 RUN STEREO FUSION (GPU FIRST, FALLBACK CPU)
    # =====================================================
    try:
        if use_gpu:
            logger.info(f"{stage}: attempting GPU stereo fusion...")
            tool_runner.run(_build_cmd(use_gpu=True), stage=stage + "_gpu")
            logger.info(f"{stage}: ✅ GPU stereo fusion successful")
        else:
            raise RuntimeError("GPU disabled, using CPU fallback")
    except Exception as e:
        logger.warning(f"{stage}: GPU execution failed ({e}), falling back to CPU...")
        tool_runner.run(_build_cmd(use_gpu=False), stage=stage + "_cpu")
        logger.info(f"{stage}: ✅ CPU stereo fusion successful")

    # =====================================================
    # 🔥 VALIDATION & METRICS (~1M points)
    # =====================================================
    if not output_path.exists():
        raise RuntimeError(f"{stage}: fusion failed → no fused PLY generated")

    pcd = o3d.io.read_point_cloud(str(output_path))
    num_points = len(pcd.points)

    logger.info(f"{stage}: fused points = {num_points}")

    if num_points < 400_000:
        status = "low"
        logger.warning(f"{stage}: LOW density → may need refinement")
    elif num_points > 1_200_000:
        status = "too_high"
        logger.warning(f"{stage}: HIGH density → may impact performance")
    else:
        status = "good"
        logger.info(f"{stage}: GOOD density → target achieved (~1M points)")

    logger.info(f"{stage}: SUCCESS → fused point cloud ready")

    return {
        "num_points": int(num_points),
        "status": status,
        "connectivity": float(connectivity),
        "avg_degree": float(avg_degree),
    }