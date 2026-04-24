from pathlib import Path
import shutil
import numpy as np


# =====================================================
# SCENE ANALYSIS (DETERMINISTIC)
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
# PARAM BUILDERS
# =====================================================

# -------- PASS 1: STABILITY (STRICT, LOW VARIANCE) ----
def _build_stable_params():
    return {
        "max_image_size": 1600,

        "window_radius": 4,
        "window_step": 1,
        "num_samples": 10,
        "num_iterations": 4,

        "sigma_spatial": -1,
        "sigma_color": 0.2,
        "ncc_sigma": 0.5,

        "min_triangulation_angle": 6,
        "incident_angle_sigma": 0.8,

        "geom_consistency": 1,
        "geom_consistency_regularizer": 0.7,
        "geom_consistency_max_cost": 1,

        # STRICT FILTERING
        "filter": 1,
        "filter_min_ncc": 0.2,
        "filter_min_triangulation_angle": 5,
        "filter_min_num_consistent": 3,
        "filter_geom_consistency_max_cost": 1,

        "cache_size": 32,
    }


# -------- PASS 2: DENSIFICATION (CONTROLLED RELAX) ----
def _build_dense_params(scene_type):
    base = {
        "max_image_size": 2000,

        "window_radius": 5,
        "window_step": 1,
        "num_samples": 14,
        "num_iterations": 4,

        "sigma_spatial": -1,
        "sigma_color": 0.25,
        "ncc_sigma": 0.6,

        "min_triangulation_angle": 4,
        "incident_angle_sigma": 0.9,

        "geom_consistency": 1,
        "geom_consistency_regularizer": 0.5,
        "geom_consistency_max_cost": 2,

        # RELAXED BUT SAFE
        "filter": 1,
        "filter_min_ncc": 0.14,
        "filter_min_triangulation_angle": 3,
        "filter_min_num_consistent": 2,
        "filter_geom_consistency_max_cost": 2,

        "cache_size": 64,
    }

    if scene_type == "turntable_object":
        base["window_radius"] = 5
    else:
        base["window_radius"] = 6

    return base


# =====================================================
# COMMAND BUILDER
# =====================================================
def _build_cmd(dense_dir, params, gpu=True, threads=None):
    cmd = [
        "colmap", "patch_match_stereo",
        "--workspace_path", str(dense_dir),
        "--workspace_format", "COLMAP",
        "--PatchMatchStereo.gpu_index", "0" if gpu else "-1",
    ]

    if not gpu and threads:
        cmd += ["--PatchMatchStereo.num_threads", str(threads)]

    for k, v in params.items():
        cmd.append(f"--PatchMatchStereo.{k}")
        cmd.append(str(v))

    return cmd


# =====================================================
# DEPTH METRIC
# =====================================================
def _load_depth(path):
    try:
        data = np.fromfile(path, dtype=np.float32)
        return data if data.size > 0 else None
    except:
        return None


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
# EXECUTION WRAPPER
# =====================================================
def _run_patchmatch(tool_runner, cmd, stage, logger):
    logger.info(f"Running: {stage}")
    tool_runner.run(cmd, stage=stage)


# =====================================================
# MAIN
# =====================================================
def run(paths, config, logger, tool_runner):
    stage = "patch_match_dual_pass"
    logger.info(f"==== {stage.upper()} ====")

    dense_dir = paths.dense
    _ensure_dense_sparse(paths, logger)

    num_images = len(list(paths.images.glob("*")))
    scene_type = _analyze_scene(num_images)

    logger.info(f"Images: {num_images}")
    logger.info(f"Scene: {scene_type}")

    depth_dir = dense_dir / "stereo" / "depth_maps"

    # =================================================
    # PASS 1 — STABILITY (CPU, DETERMINISTIC BASE)
    # =================================================
    stable_params = _build_stable_params()

    try:
        _run_patchmatch(
            tool_runner,
            _build_cmd(dense_dir, stable_params, gpu=False, threads=1),
            stage + "_pass1_stable_cpu",
            logger
        )
    except Exception as e:
        logger.warning(f"Stable pass CPU failed → GPU fallback: {e}")

        _run_patchmatch(
            tool_runner,
            _build_cmd(dense_dir, stable_params, gpu=True),
            stage + "_pass1_stable_gpu",
            logger
        )

    coverage1 = _compute_coverage(depth_dir)
    logger.info(f"[PASS 1] Coverage: {coverage1:.2f}%")

    # =================================================
    # PASS 2 — DENSIFICATION (GPU REFINEMENT)
    # =================================================
    dense_params = _build_dense_params(scene_type)

    _run_patchmatch(
        tool_runner,
        _build_cmd(dense_dir, dense_params, gpu=True),
        stage + "_pass2_dense_gpu",
        logger
    )

    coverage2 = _compute_coverage(depth_dir)
    logger.info(f"[PASS 2] Coverage: {coverage2:.2f}%")

    # =================================================
    # QUALITY LOGIC
    # =================================================
    if coverage2 < coverage1:
        logger.warning("⚠️ Densification reduced quality → check params")

    if coverage2 > 60:
        logger.info("High density achieved (ready for pruning stage)")

    return {
        "status": "complete",
        "pass1_coverage": coverage1,
        "pass2_coverage": coverage2,
        "images": num_images,
        "scene_type": scene_type
    }