from pathlib import Path
import shutil
import numpy as np


# =====================================================
# VALIDATE SPARSE MODEL
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
# BASE PARAMS
# =====================================================
def _base_params():
    return {
        "max_img_size": 1800,
        "window_radius": 4,
        "num_samples": 12,
        "num_iterations": 3,
        "geom_consistency": 1,
        "geom_consistency_regularizer": 0.3,
        "filter_min_ncc": 0.12,
        "filter_min_num_consistent": 2,
        "min_triangulation_angle": 2.5,
        "sigma_spatial": 4.0,
        "sigma_color": 0.2,
        "cache_size": 32,
    }


# =====================================================
# PARAM ADJUSTMENT
# =====================================================
def _adjust_params(base):
    p = base.copy()

    # generalized "recovery mode"
    p["filter_min_ncc"] = 0.08
    p["num_samples"] += 6
    p["window_radius"] += 1
    p["num_iterations"] += 2

    return p


# =====================================================
# BUILD COMMAND
# =====================================================
def _build_cmd(dense_dir, p, gpu=True):
    return [
        "colmap", "patch_match_stereo",
        "--workspace_path", str(dense_dir),
        "--workspace_format", "COLMAP",

        "--PatchMatchStereo.gpu_index", "0" if gpu else "-1",
        "--PatchMatchStereo.max_image_size", str(p["max_img_size"]),

        "--PatchMatchStereo.window_radius", str(p["window_radius"]),
        "--PatchMatchStereo.num_samples", str(p["num_samples"]),
        "--PatchMatchStereo.num_iterations", str(p["num_iterations"]),

        "--PatchMatchStereo.sigma_spatial", str(p["sigma_spatial"]),
        "--PatchMatchStereo.sigma_color", str(p["sigma_color"]),

        "--PatchMatchStereo.geom_consistency", str(p["geom_consistency"]),
        "--PatchMatchStereo.geom_consistency_regularizer",
        str(p["geom_consistency_regularizer"]),

        "--PatchMatchStereo.min_triangulation_angle",
        str(p["min_triangulation_angle"]),

        "--PatchMatchStereo.filter", "1",
        "--PatchMatchStereo.filter_min_ncc", str(p["filter_min_ncc"]),
        "--PatchMatchStereo.filter_min_num_consistent",
        str(p["filter_min_num_consistent"]),

        "--PatchMatchStereo.cache_size", str(p["cache_size"]),
    ]


# =====================================================
# LOAD DEPTH MAP
# =====================================================
def _load_depth(path):
    try:
        data = np.fromfile(path, dtype=np.float32)
        return data if data.size > 0 else None
    except:
        return None


# =====================================================
# ANALYZE IMAGE QUALITY
# =====================================================
def _analyze_images(depth_dir: Path):
    results = {}

    for f in depth_dir.glob("*.bin"):
        data = _load_depth(f)
        if data is None:
            results[f.stem] = ("FAIL", 0)
            continue

        valid = np.count_nonzero(data > 0)
        ratio = valid / data.size

        if ratio > 0.65:
            results[f.stem] = ("GOOD", ratio)
        else:
            results[f.stem] = ("WEAK", ratio)

    return results


# =====================================================
# BUILD GLOBAL COVERAGE MASK
# =====================================================
def _build_coverage(depth_dir: Path, strong_images):
    coverage = None

    for img in strong_images:
        f = depth_dir / f"{img}.bin"
        data = _load_depth(f)

        if data is None:
            continue

        mask = (data > 0).astype(np.uint8)

        if coverage is None:
            coverage = mask
        else:
            coverage = np.maximum(coverage, mask)

    return coverage


# =====================================================
# CHECK IF IMAGE ADDS NEW GEOMETRY
# =====================================================
def _adds_new_geometry(depth_dir, img, coverage):
    f = depth_dir / f"{img}.bin"
    data = _load_depth(f)

    if data is None or coverage is None:
        return False

    mask = (data > 0).astype(np.uint8)

    new_pixels = np.logical_and(mask == 1, coverage == 0)
    gain = np.count_nonzero(new_pixels) / mask.size

    return gain > 0.05  # threshold


# =====================================================
# DELETE OUTPUTS
# =====================================================
def _delete_outputs(dense_dir, images):
    depth_dir = dense_dir / "stereo" / "depth_maps"
    normal_dir = dense_dir / "stereo" / "normal_maps"

    for img in images:
        base = img.replace(".JPG", "").replace(".jpg", "")

        for f in depth_dir.glob(f"{base}*"):
            f.unlink(missing_ok=True)

        for f in normal_dir.glob(f"{base}*"):
            f.unlink(missing_ok=True)


# =====================================================
# MAIN
# =====================================================
def run(paths, config, logger, tool_runner):
    stage = "patch_match_geometry_aware"
    logger.info(f"---- {stage.upper()} ----")

    dense_dir = paths.dense
    _ensure_dense_sparse(paths, logger)

    base = _base_params()

    # -------------------------------------------------
    # PASS 1
    # -------------------------------------------------
    logger.info("Running full PatchMatch...")

    try:
        tool_runner.run(_build_cmd(dense_dir, base, True), stage=stage + "_gpu")
    except Exception:
        tool_runner.run(_build_cmd(dense_dir, base, False), stage=stage + "_cpu")

    depth_dir = dense_dir / "stereo" / "depth_maps"

    # -------------------------------------------------
    # ANALYZE
    # -------------------------------------------------
    analysis = _analyze_images(depth_dir)

    strong = []
    weak = []

    print("\n=== IMAGE QUALITY ===")

    for img, (status, score) in analysis.items():
        print(f"{img}: {status} ({score:.2f})")

        normalized = img.split(".")[0]

        if status == "GOOD":
            strong.append(normalized)
        else:
            weak.append(normalized)

    # -------------------------------------------------
    # BUILD GEOMETRY FROM STRONG
    # -------------------------------------------------
    logger.info(f"Building geometry from {len(strong)} strong images...")
    coverage = _build_coverage(depth_dir, strong)

    # -------------------------------------------------
    # SELECT USEFUL WEAK IMAGES
    # -------------------------------------------------
    selected = []

    for img in weak:
        if _adds_new_geometry(depth_dir, img, coverage):
            selected.append(img)

    logger.info(f"Selected {len(selected)} weak images that add geometry")

    # -------------------------------------------------
    # RE-RUN ONLY IF USEFUL
    # -------------------------------------------------
    if selected:
        _delete_outputs(dense_dir, selected)

        refined = _adjust_params(base)

        logger.info("Re-running PatchMatch for geometry completion...")

        try:
            tool_runner.run(
                _build_cmd(dense_dir, refined, True),
                stage=stage + "_refine_gpu"
            )
        except Exception:
            tool_runner.run(
                _build_cmd(dense_dir, refined, False),
                stage=stage + "_refine_cpu"
            )

    # -------------------------------------------------
    # FINAL SCORE
    # -------------------------------------------------
    total_valid = 0
    total_pixels = 0

    for f in depth_dir.glob("*.bin"):
        data = _load_depth(f)
        if data is None:
            continue

        total_valid += np.count_nonzero(data > 0)
        total_pixels += data.size

    score = (total_valid / total_pixels) * 100 if total_pixels else 0

    print("\n=== FINAL SCORE ===")
    print(f"Depth Coverage: {score:.2f}%")

    return {
        "status": "geometry_aware_complete",
        "quality_score": score,
        "strong_images": len(strong),
        "used_weak_images": len(selected)
    }