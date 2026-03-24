from pathlib import Path


# =====================================================
# 🔍 VALIDATION HELPERS
# =====================================================

def _is_valid_model_dir(p: Path):
    """Check if directory contains a valid COLMAP sparse model."""
    return (
        (p / "cameras.bin").exists()
        and (p / "images.bin").exists()
        and (p / "points3D.bin").exists()
    )


def _find_best_sparse_model(sparse_root: Path, logger):
    """
    Detect and select the best sparse model.

    Supports:
    - flat layout: sparse/
    - multi-model: sparse/0, sparse/1, ...

    Selection heuristic:
    - largest points3D.bin (most complete reconstruction)
    """

    # Case 1: flat structure
    if _is_valid_model_dir(sparse_root):
        logger.info("openmvs_export: using flat sparse model")
        return sparse_root

    # Case 2: multiple submodels
    candidates = []

    for sub in sparse_root.iterdir():
        if sub.is_dir() and _is_valid_model_dir(sub):
            points_file = sub / "points3D.bin"
            size = points_file.stat().st_size
            candidates.append((sub, size))

    if not candidates:
        raise RuntimeError("openmvs_export: no valid sparse models found")

    # Select best model (largest reconstruction)
    best_model, best_size = max(candidates, key=lambda x: x[1])

    logger.info(
        f"openmvs_export: selected model → {best_model.name} "
        f"(points3D.bin size={best_size})"
    )

    return best_model


# =====================================================
# 🚀 MAIN STAGE
# =====================================================

def run(paths, config, logger, tool_runner):
    stage = "openmvs_export"
    logger.info(f"---- {stage.upper()} ----")

    sparse_root = paths.sparse

    # 🔥 IMPORTANT: use UNDISTORTED images
    image_dir = paths.working / "images"

    if not sparse_root.exists():
        raise RuntimeError(f"{stage}: sparse folder missing → {sparse_root}")

    if not image_dir.exists() or not any(image_dir.iterdir()):
        raise RuntimeError(f"{stage}: undistorted images missing → {image_dir}")

    # =====================================================
    # 🔍 FIND BEST SPARSE MODEL
    # =====================================================
    sparse_model = _find_best_sparse_model(sparse_root, logger)

    # =====================================================
    # 📁 OUTPUT DIR (OpenMVS workspace)
    # =====================================================
    mvs_dir = paths.run_root / "openmvs"
    mvs_dir.mkdir(parents=True, exist_ok=True)

    scene_file = mvs_dir / "scene.mvs"

    if scene_file.exists():
        logger.warning(f"{stage}: removing old scene.mvs")
        scene_file.unlink()

    logger.info(f"{stage}: exporting COLMAP → OpenMVS")

    # =====================================================
    # 🔥 CORRECT COMMAND (FIXED)
    # =====================================================
    cmd = [
        "InterfaceCOLMAP",

        # COLMAP sparse model (REQUIRED)
        "-i", str(sparse_model),

        # 🔥 CORRECT FLAG (capital I)
        "-I", str(image_dir),

        # output
        "-o", str(scene_file),
        "-w", str(mvs_dir),
    ]

    logger.info(f"[{stage}] COMMAND:\n{' '.join(cmd)}")

    tool_runner.run(cmd, stage=stage)

    # =====================================================
    # ✅ VALIDATION
    # =====================================================
    if not scene_file.exists():
        raise RuntimeError(f"{stage}: scene.mvs not created")

    logger.info(f"{stage}: SUCCESS → {scene_file}")