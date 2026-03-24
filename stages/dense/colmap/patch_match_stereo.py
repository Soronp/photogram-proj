from pathlib import Path
import shutil


def _validate_sparse_model(sparse_dir: Path):
    """
    Ensure sparse_dir contains a valid COLMAP model.
    Handles both direct and '0/' subfolder layouts.
    """
    if not sparse_dir.exists():
        return False, None

    # Case 1: direct files
    if (sparse_dir / "cameras.bin").exists():
        return True, sparse_dir

    # Case 2: inside subfolder (0/)
    subfolders = [p for p in sparse_dir.iterdir() if p.is_dir()]
    for sf in subfolders:
        if (sf / "cameras.bin").exists():
            return True, sf

    return False, None


def _ensure_dense_sparse(paths, logger):
    """
    Guarantees dense/sparse contains a valid model.
    Copies from paths.sparse if needed.
    """
    dense_sparse = paths.dense / "sparse"

    valid, actual_path = _validate_sparse_model(dense_sparse)

    if valid:
        logger.info(f"sparse model valid at {actual_path}")
        return actual_path

    logger.warning("dense/sparse invalid → rebuilding from sparse stage")

    if not paths.sparse.exists():
        raise RuntimeError("Global sparse model missing → upstream failure")

    # Reset
    if dense_sparse.exists():
        shutil.rmtree(dense_sparse)

    shutil.copytree(paths.sparse, dense_sparse)

    # Re-validate
    valid, actual_path = _validate_sparse_model(dense_sparse)

    if not valid:
        raise RuntimeError("Failed to create valid dense/sparse model")

    logger.info(f"sparse model rebuilt at {actual_path}")
    return actual_path


def run(paths, config, logger, tool_runner):
    stage = "patch_match_stereo"
    logger.info(f"---- {stage.upper()} ----")

    dense_dir = paths.dense
    images_dir = dense_dir / "images"

    if not images_dir.exists() or not any(images_dir.iterdir()):
        raise RuntimeError(f"{stage}: undistorted images missing")

    # 🔥 CRITICAL FIX
    sparse_dir = _ensure_dense_sparse(paths, logger)

    retry = config.get("_meta", {}).get("retry_count", 0)

    # =====================================================
    # QUALITY TIERS (unchanged)
    # =====================================================
    if retry == 0:
        max_img_size = 2600
        window_radius = 6
        num_samples = 20
        num_iterations = 6
    elif retry == 1:
        max_img_size = 3000
        window_radius = 7
        num_samples = 24
        num_iterations = 7
    else:
        max_img_size = 3200
        window_radius = 8
        num_samples = 28
        num_iterations = 8

    logger.info(
        f"{stage}: size={max_img_size}, radius={window_radius}, "
        f"samples={num_samples}, iters={num_iterations}"
    )

    # =====================================================
    # STABLE PARAMETERS
    # =====================================================
    params = {
        "min_triangulation_angle": 0.6,
        "filter_min_triangulation_angle": 1.0,
        "filter_min_ncc": 0.06,
        "geom_consistency": 1,
        "consistency_regularizer": 0.25,
        "filter_min_num_consistent": 2,
    }

    def _build_cmd(use_gpu=True):
        return [
            "colmap", "patch_match_stereo",
            "--workspace_path", str(dense_dir),
            "--workspace_format", "COLMAP",

            "--PatchMatchStereo.max_image_size", str(max_img_size),
            "--PatchMatchStereo.gpu_index", "0" if use_gpu else "-1",

            "--PatchMatchStereo.window_radius", str(window_radius),
            "--PatchMatchStereo.num_samples", str(num_samples),
            "--PatchMatchStereo.num_iterations", str(num_iterations),

            "--PatchMatchStereo.min_triangulation_angle", str(params["min_triangulation_angle"]),
            "--PatchMatchStereo.filter_min_triangulation_angle", str(params["filter_min_triangulation_angle"]),
            "--PatchMatchStereo.filter_min_ncc", str(params["filter_min_ncc"]),

            "--PatchMatchStereo.geom_consistency", str(params["geom_consistency"]),
            "--PatchMatchStereo.geom_consistency_regularizer", str(params["consistency_regularizer"]),

            "--PatchMatchStereo.filter", "1",
            "--PatchMatchStereo.filter_min_num_consistent", str(params["filter_min_num_consistent"]),

            "--PatchMatchStereo.cache_size", "128",
        ]

    try:
        logger.info(f"{stage}: running GPU...")
        tool_runner.run(_build_cmd(True), stage=stage + "_gpu")
    except Exception as e:
        logger.warning(f"{stage}: GPU failed → {e}")
        tool_runner.run(_build_cmd(False), stage=stage + "_cpu")

    logger.info(f"{stage}: SUCCESS")