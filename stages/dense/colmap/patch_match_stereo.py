from pathlib import Path
import shutil
import numpy as np


# =====================================================
# SPARSE VALIDATION
# =====================================================
def _validate_sparse_model(sparse_dir: Path):
    if not sparse_dir.exists():
        return False, None

    if (sparse_dir / "cameras.bin").exists():
        return True, sparse_dir

    for sf in sparse_dir.iterdir():
        if sf.is_dir() and (sf / "cameras.bin").exists():
            return True, sf

    return False, None


# =====================================================
# ENSURE CONSISTENCY
# =====================================================
def _ensure_dense_sparse(paths, logger):
    dense_sparse = paths.dense / "sparse"

    valid, actual = _validate_sparse_model(dense_sparse)
    if valid:
        return actual

    logger.warning("Sparse missing → restoring")

    if dense_sparse.exists():
        shutil.rmtree(dense_sparse)

    shutil.copytree(paths.sparse, dense_sparse)
    return dense_sparse


# =====================================================
# ADAPTIVE PARAMS (STRICT TO CLI)
# =====================================================
def _build_params(num_images):
    """
    Tuned for:
    - high density
    - controlled runtime
    - stable geometry
    """

    # Resolution scaling
    if num_images < 40:
        max_img_size = 2000
    elif num_images < 100:
        max_img_size = 1800
    else:
        max_img_size = 1600

def _build_params(num_images):
    if num_images < 40:
        max_img_size = 1800
    elif num_images < 100:
        max_img_size = 1600
    else:
        max_img_size = 1400

    return {
        "max_img_size": max_img_size,

        # 🔥 BIG SPEED GAIN
        "window_radius": 5,
        "window_step": 2,   # was 1 → halves compute

        # 🔥 MAJOR SPEED CONTROL
        "num_samples": 20,  # was ~40
        "num_iterations": 5,  # was ~8

        # matching
        "sigma_spatial": 4.0,
        "sigma_color": 0.25,
        "ncc_sigma": 0.6,

        # geometry
        "min_triangulation_angle": 1.0,
        "incident_angle_sigma": 1.0,

        # 🔥 KEEP THIS ON
        "geom_consistency": 1,
        "geom_consistency_regularizer": 0.35,
        "geom_consistency_max_cost": 3,

        # 🔥 ARTIFACT CONTROL (IMPORTANT)
        "filter_min_ncc": 0.05,   # was 0.02 → cleaner
        "filter_min_num_consistent": 2,
        "filter_min_triangulation_angle": 1.0,
        "filter_geom_consistency_max_cost": 2,

        "cache_size": 64,
    }


# =====================================================
# COMMAND BUILDER (STRICT CLI FLAGS)
# =====================================================
def _build_cmd(dense_dir, p, gpu=True):
    return [
        "colmap", "patch_match_stereo",

        "--workspace_path", str(dense_dir),
        "--workspace_format", "COLMAP",

        "--PatchMatchStereo.gpu_index", "0" if gpu else "-1",

        "--PatchMatchStereo.max_image_size", str(p["max_img_size"]),

        "--PatchMatchStereo.window_radius", str(p["window_radius"]),
        "--PatchMatchStereo.window_step", str(p["window_step"]),

        "--PatchMatchStereo.num_samples", str(p["num_samples"]),
        "--PatchMatchStereo.num_iterations", str(p["num_iterations"]),

        "--PatchMatchStereo.sigma_spatial", str(p["sigma_spatial"]),
        "--PatchMatchStereo.sigma_color", str(p["sigma_color"]),
        "--PatchMatchStereo.ncc_sigma", str(p["ncc_sigma"]),

        "--PatchMatchStereo.min_triangulation_angle",
        str(p["min_triangulation_angle"]),

        "--PatchMatchStereo.incident_angle_sigma",
        str(p["incident_angle_sigma"]),

        "--PatchMatchStereo.geom_consistency",
        str(p["geom_consistency"]),

        "--PatchMatchStereo.geom_consistency_regularizer",
        str(p["geom_consistency_regularizer"]),

        "--PatchMatchStereo.geom_consistency_max_cost",
        str(p["geom_consistency_max_cost"]),

        "--PatchMatchStereo.filter", "1",

        "--PatchMatchStereo.filter_min_ncc",
        str(p["filter_min_ncc"]),

        "--PatchMatchStereo.filter_min_num_consistent",
        str(p["filter_min_num_consistent"]),

        "--PatchMatchStereo.filter_min_triangulation_angle",
        str(p["filter_min_triangulation_angle"]),

        "--PatchMatchStereo.filter_geom_consistency_max_cost",
        str(p["filter_geom_consistency_max_cost"]),

        "--PatchMatchStereo.cache_size", str(p["cache_size"]),
    ]


# =====================================================
# DEPTH LOADING
# =====================================================
def _load_depth(path):
    try:
        data = np.fromfile(path, dtype=np.float32)
        return data if data.size > 0 else None
    except:
        return None


# =====================================================
# COVERAGE COMPUTATION
# =====================================================
def _compute_coverage(depth_dir: Path):
    total_valid = 0
    total_pixels = 0

    for f in depth_dir.glob("*.bin"):
        data = _load_depth(f)
        if data is None:
            continue

        total_valid += np.count_nonzero(data > 0)
        total_pixels += data.size

    return (total_valid / total_pixels) * 100 if total_pixels else 0


# =====================================================
# MAIN PIPELINE (SINGLE PASS, GPU→CPU)
# =====================================================
def run(paths, config, logger, tool_runner):
    stage = "patch_match_single_pass"
    logger.info(f"---- {stage.upper()} ----")

    dense_dir = paths.dense
    _ensure_dense_sparse(paths, logger)

    # -------------------------------------------------
    # DATASET SIZE
    # -------------------------------------------------
    num_images = len(list(paths.images.glob("*")))
    logger.info(f"Images detected: {num_images}")

    # -------------------------------------------------
    # PARAM BUILD
    # -------------------------------------------------
    params = _build_params(num_images)
    logger.info(f"Params: {params}")

    # -------------------------------------------------
    # RUN (GPU FIRST → CPU FALLBACK)
    # -------------------------------------------------
    try:
        logger.info("Running PatchMatch (GPU)")
        tool_runner.run(
            _build_cmd(dense_dir, params, gpu=True),
            stage=stage + "_gpu"
        )
    except Exception as e:
        logger.warning(f"GPU failed → fallback to CPU: {e}")

        tool_runner.run(
            _build_cmd(dense_dir, params, gpu=False),
            stage=stage + "_cpu"
        )

    # -------------------------------------------------
    # EVALUATION
    # -------------------------------------------------
    depth_dir = dense_dir / "stereo" / "depth_maps"
    score = _compute_coverage(depth_dir)

    print("\n=== FINAL SCORE ===")
    print(f"Depth Coverage: {score:.2f}%")

    return {
        "status": "complete",
        "quality_score": score,
        "images": num_images
    }