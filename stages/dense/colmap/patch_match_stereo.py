from pathlib import Path
import shutil
import numpy as np


# =====================================================
# SCENE ANALYSIS (KEEP SIMPLE, DETERMINISTIC)
# =====================================================
def _analyze_scene(num_images):
    return "turntable_object" if num_images <= 60 else "generic"


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
# ENSURE DENSE/SPARSE CONSISTENCY
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
# PARAM BUILDERS (DENSIFICATION-OPTIMIZED)
# =====================================================
def _build_turntable_params():
    return {
        # -------------------------------
        # IMAGE HANDLING
        # -------------------------------
        "max_image_size": 2000,

        # -------------------------------
        # CORE PATCHMATCH (KEY DENSITY LEVER)
        # -------------------------------
        "window_radius": 5,        # ⬆ better surface continuity
        "window_step": 1,
        "num_samples": 18,         # ⬆ more hypotheses (MAIN densifier)
        "num_iterations": 5,       # keep bounded

        # -------------------------------
        # PHOTOMETRIC MATCHING
        # -------------------------------
        "sigma_spatial": -1,
        "sigma_color": 0.25,       # slightly more tolerant
        "ncc_sigma": 0.55,         # allow harder surfaces

        # -------------------------------
        # GEOMETRY (DO NOT OVER-RELAX)
        # -------------------------------
        "min_triangulation_angle": 4,
        "incident_angle_sigma": 0.9,

        # -------------------------------
        # GEOMETRIC CONSISTENCY (CRITICAL)
        # -------------------------------
        "geom_consistency": 1,
        "geom_consistency_regularizer": 0.5,
        "geom_consistency_max_cost": 2,

        # -------------------------------
        # FILTERING (CONTROLLED RELAXATION)
        # -------------------------------
        "filter": 1,
        "filter_min_ncc": 0.12,                 # ⬇ admit more points
        "filter_min_triangulation_angle": 3,
        "filter_min_num_consistent": 2,         # ⬇ allow weaker agreement
        "filter_geom_consistency_max_cost": 2,

        # -------------------------------
        # PERFORMANCE
        # -------------------------------
        "cache_size": 64,
    }


def _build_generic_params():
    return {
        "max_image_size": 2000,

        "window_radius": 6,
        "window_step": 1,
        "num_samples": 18,
        "num_iterations": 5,

        "sigma_spatial": -1,
        "sigma_color": 0.3,
        "ncc_sigma": 0.6,

        "min_triangulation_angle": 2,
        "incident_angle_sigma": 1.0,

        "geom_consistency": 1,
        "geom_consistency_regularizer": 0.4,
        "geom_consistency_max_cost": 3,

        "filter": 1,
        "filter_min_ncc": 0.10,
        "filter_min_triangulation_angle": 2,
        "filter_min_num_consistent": 2,
        "filter_geom_consistency_max_cost": 2,

        "cache_size": 32,
    }


# =====================================================
# PARAM SELECTOR
# =====================================================
def _build_params(scene_type):
    return _build_turntable_params() if scene_type == "turntable_object" else _build_generic_params()


# =====================================================
# COMMAND BUILDER
# =====================================================
def _build_cmd(dense_dir, params, gpu=True):
    cmd = [
        "colmap", "patch_match_stereo",
        "--workspace_path", str(dense_dir),
        "--workspace_format", "COLMAP",
        "--PatchMatchStereo.gpu_index", "0" if gpu else "-1",
    ]

    for k, v in params.items():
        cmd.append(f"--PatchMatchStereo.{k}")
        cmd.append(str(v))

    return cmd


# =====================================================
# DEPTH LOADER
# =====================================================
def _load_depth(path):
    try:
        data = np.fromfile(path, dtype=np.float32)
        return data if data.size > 0 else None
    except:
        return None


# =====================================================
# COVERAGE METRIC (IMPORTANT FEEDBACK SIGNAL)
# =====================================================
def _compute_coverage(depth_dir: Path):
    total_valid = 0
    total_pixels = 0

    for f in depth_dir.glob("*.bin"):
        data = _load_depth(f)
        if data is None:
            continue

        valid = data > 0
        total_valid += np.count_nonzero(valid)
        total_pixels += data.size

    return (total_valid / total_pixels) * 100 if total_pixels else 0


# =====================================================
# MAIN
# =====================================================
def run(paths, config, logger, tool_runner):
    stage = "patch_match_densified_v3"
    logger.info(f"---- {stage.upper()} ----")

    dense_dir = paths.dense
    _ensure_dense_sparse(paths, logger)

    num_images = len(list(paths.images.glob("*")))
    logger.info(f"Images detected: {num_images}")

    # -----------------------------
    # SCENE TYPE
    # -----------------------------
    scene_type = _analyze_scene(num_images)
    logger.info(f"Scene type: {scene_type}")

    # -----------------------------
    # PARAMS
    # -----------------------------
    params = _build_params(scene_type)
    logger.info(f"Params: {params}")

    # -----------------------------
    # EXECUTION
    # -----------------------------
    try:
        tool_runner.run(
            _build_cmd(dense_dir, params, gpu=True),
            stage=stage + "_gpu"
        )
    except Exception as e:
        logger.warning(f"GPU failed → CPU fallback: {e}")

        tool_runner.run(
            _build_cmd(dense_dir, params, gpu=False),
            stage=stage + "_cpu"
        )

    # -----------------------------
    # QUALITY CHECK
    # -----------------------------
    depth_dir = dense_dir / "stereo" / "depth_maps"
    score = _compute_coverage(depth_dir)

    logger.info(f"Depth coverage: {score:.2f}%")

    # -----------------------------
    # SANITY FLAGGING
    # -----------------------------
    if score < 5:
        logger.warning("⚠️ Extremely low coverage → likely failure")
    elif score > 60:
        logger.info("High density achieved (good candidate for pruning)")

    return {
        "status": "complete",
        "quality_score": score,
        "images": num_images,
        "scene_type": scene_type
    }