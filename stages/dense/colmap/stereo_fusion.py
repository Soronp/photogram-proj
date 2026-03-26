from pathlib import Path
import open3d as o3d
import numpy as np


# =====================================================
# RUN FUSION
# =====================================================
def _run_fusion(tool_runner, dense_dir, out, p):
    tool_runner.run([
        "colmap", "stereo_fusion",
        "--workspace_path", str(dense_dir),
        "--workspace_format", "COLMAP",
        "--input_type", "geometric",
        "--output_path", str(out),

        "--StereoFusion.min_num_pixels", str(p["min_pixels"]),
        "--StereoFusion.max_reproj_error", str(p["max_reproj"]),
        "--StereoFusion.max_depth_error", str(p["max_depth"]),
        "--StereoFusion.max_normal_error", str(p["max_normal"]),
        "--StereoFusion.max_traversal_depth", str(p["traversal"]),
        "--StereoFusion.num_threads", "1",
    ])


# =====================================================
# LOAD POINT CLOUD
# =====================================================
def _load(path):
    pcd = o3d.io.read_point_cloud(str(path))
    return pcd if len(pcd.points) > 0 else None


# =====================================================
# SAFE POST-PROCESS (NON-DESTRUCTIVE)
# =====================================================
def _safe_post(pcd):
    if pcd is None:
        return None

    original_count = len(pcd.points)

    # VERY light denoise only
    filtered, ind = pcd.remove_statistical_outlier(
        nb_neighbors=8,
        std_ratio=3.0
    )

    # If we removed too many points → reject
    if len(filtered.points) < 0.85 * original_count:
        return pcd  # keep original

    filtered.estimate_normals()
    return filtered


# =====================================================
# QUALITY METRIC
# =====================================================
def _evaluate(pcd):
    if pcd is None or len(pcd.points) == 0:
        return 0.0

    pts = np.asarray(pcd.points)

    bbox = pts.max(axis=0) - pts.min(axis=0)
    volume = np.prod(bbox) if np.all(bbox > 0) else 1

    density = len(pts) / volume

    return min(100, density * 0.05)


# =====================================================
# PARAM POLICY (KEEP STRICT BY DEFAULT)
# =====================================================
def _build_params(retry):
    if retry == 0:
        return {
            "min_pixels": 3,
            "max_reproj": 2.0,
            "max_depth": 0.02,
            "max_normal": 18,
            "traversal": 80,
        }
    elif retry == 1:
        return {
            "min_pixels": 2,
            "max_reproj": 2.5,
            "max_depth": 0.025,
            "max_normal": 22,
            "traversal": 100,
        }
    else:
        return {
            "min_pixels": 2,
            "max_reproj": 3.0,
            "max_depth": 0.03,
            "max_normal": 25,
            "traversal": 120,
        }


# =====================================================
# MAIN
# =====================================================
def run(paths, config, logger, tool_runner):
    stage = "stereo_fusion_correct"
    logger.info(f"==== {stage.upper()} ====")

    dense = paths.dense
    retry = config.get("_meta", {}).get("retry_count", 0)

    params = _build_params(retry)

    raw_path = dense / "fused_raw.ply"

    logger.info(
        f"Fusion → pixels={params['min_pixels']} | "
        f"reproj={params['max_reproj']} | "
        f"depth={params['max_depth']}"
    )

    # -------------------------------------------------
    # RUN FUSION
    # -------------------------------------------------
    _run_fusion(tool_runner, dense, raw_path, params)

    raw_pcd = _load(raw_path)
    raw_score = _evaluate(raw_pcd)

    # -------------------------------------------------
    # SAFE POST PROCESS
    # -------------------------------------------------
    processed_pcd = _safe_post(raw_pcd)
    processed_score = _evaluate(processed_pcd)

    # -------------------------------------------------
    # SELECT BEST VERSION
    # -------------------------------------------------
    if processed_score >= raw_score:
        final_pcd = processed_pcd
        final_score = processed_score
        logger.info("Using processed point cloud")
    else:
        final_pcd = raw_pcd
        final_score = raw_score
        logger.info("Using raw point cloud (better geometry preserved)")

    # -------------------------------------------------
    # OPTIONAL RETRY (ONLY IF VERY LOW)
    # -------------------------------------------------
    if final_score < 40:
        logger.warning("Low fusion quality → retrying relaxed fusion")

        retry_params = _build_params(retry + 1)
        retry_path = dense / "fused_retry.ply"

        _run_fusion(tool_runner, dense, retry_path, retry_params)

        retry_pcd = _load(retry_path)
        retry_score = _evaluate(retry_pcd)

        if retry_score > final_score:
            final_pcd = retry_pcd
            final_score = retry_score
            logger.info("Retry improved result")

    # -------------------------------------------------
    # SAVE FINAL
    # -------------------------------------------------
    final_path = dense / "fused.ply"
    o3d.io.write_point_cloud(str(final_path), final_pcd)

    # -------------------------------------------------
    # OUTPUT
    # -------------------------------------------------
    print("\n=== FUSION QUALITY SCORE ===")
    print(f"Point Density Score: {final_score:.2f}%")

    if final_score > 70:
        print("Quality: HIGH")
    elif final_score > 45:
        print("Quality: MEDIUM")
    else:
        print("Quality: LOW")

    logger.info(f"Final points: {len(final_pcd.points)}")

    return {
        "status": "fusion_complete",
        "quality_score": final_score,
        "points": len(final_pcd.points),
        "used_raw": final_pcd is raw_pcd
    }