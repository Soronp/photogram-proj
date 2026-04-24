from pathlib import Path
import shutil
import subprocess


# =====================================================
# 🔍 COLMAP / GLOMAP MODEL CHECK
# =====================================================
def _analyze_colmap_model(model_path: Path):
    images = model_path / "images.bin"
    points = model_path / "points3D.bin"

    if not images.exists():
        return False, 0

    pts_count = 0

    if points.exists():
        try:
            cmd = ["colmap", "model_analyzer", "--path", str(model_path)]
            result = subprocess.run(cmd, capture_output=True, text=True)

            for line in result.stdout.splitlines():
                if "Points:" in line:
                    pts_count = int(line.split(":")[-1].strip())
                    break
        except:
            pts_count = 0

    return True, pts_count


# =====================================================
# 🔍 FIND BEST MODEL (COLMAP/GLOMAP)
# =====================================================
def _find_best_colmap_model(sparse_root: Path, logger):
    models = [p for p in sparse_root.iterdir() if p.is_dir()]

    if not models:
        raise RuntimeError("No sparse models found")

    scored = []

    for m in models:
        valid, pts = _analyze_colmap_model(m)
        if valid:
            scored.append((m, pts))

    if not scored:
        raise RuntimeError("No usable model (missing images.bin)")

    best = max(scored, key=lambda x: x[1])

    if best[1] < 50:
        logger.warning(
            f"weak sparse model ({best[1]} points) → results may degrade"
        )

    return best[0]


# =====================================================
# 🔍 FIND OPENMVG MODEL
# =====================================================
def _find_openmvg_model(sparse_root: Path):
    sfm_file = sparse_root / "openmvg_reconstruction" / "sfm_data.bin"

    if not sfm_file.exists():
        raise RuntimeError("OpenMVG sfm_data.bin not found")

    return sfm_file


# =====================================================
# 🧹 CLEAN DENSE
# =====================================================
def _clean_dense(dense_dir: Path, logger):
    for sub in ["images", "sparse"]:
        p = dense_dir / sub
        if p.exists():
            logger.warning(f"clearing {p}")
            shutil.rmtree(p)


# =====================================================
# 🚀 MAIN
# =====================================================
def run(paths, config, logger, tool_runner):
    stage = "image_undistorter"
    logger.info(f"---- {stage.upper()} ----")

    sparse_root = paths.sparse
    image_dir = paths.images
    dense_dir = paths.dense

    backend = config.get("pipeline", {}).get("backends", {}).get("sparse")

    if not backend:
        backend = "colmap"

    # =====================================================
    # 🔥 VALIDATION
    # =====================================================
    if not sparse_root.exists():
        raise RuntimeError("Sparse folder missing")

    if not image_dir.exists():
        raise RuntimeError("Image directory missing")

    dense_dir.mkdir(parents=True, exist_ok=True)

    # =====================================================
    # 🔵 COLMAP / GLOMAP (UNIFIED PATH)
    # =====================================================
    if backend in ("colmap", "glomap"):
        logger.info(f"image_undistorter: backend = {backend.upper()}")

        sparse_model = _find_best_colmap_model(sparse_root, logger)
        logger.info(f"Using model → {sparse_model.name}")

        _clean_dense(dense_dir, logger)

        cmd = [
            "colmap", "image_undistorter",
            "--image_path", str(image_dir),
            "--input_path", str(sparse_model),
            "--output_path", str(dense_dir),
            "--output_type", "COLMAP",
        ]

        tool_runner.run(cmd, stage=stage)

    # =====================================================
    # 🟢 OPENMVG
    # =====================================================
    elif backend == "openmvg":
        logger.info("image_undistorter: backend = OPENMVG")

        sfm_file = _find_openmvg_model(sparse_root)
        logger.info(f"Using sfm_data → {sfm_file}")

        _clean_dense(dense_dir, logger)

        undistorted_dir = dense_dir / "images"
        undistorted_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "openMVG_main_ExportUndistortedImages",
            "-i", str(sfm_file),
            "-o", str(undistorted_dir)
        ]

        tool_runner.run(cmd, stage=stage)

    else:
        raise ValueError(f"Unsupported sparse backend: {backend}")

    # =====================================================
    # ✅ VALIDATION
    # =====================================================
    out_images = list((dense_dir / "images").glob("*"))
    in_images = list(image_dir.glob("*"))

    if not out_images:
        raise RuntimeError("Undistortion failed: no output images")

    coverage = len(out_images) / max(len(in_images), 1)

    logger.info(f"{stage}: output images = {len(out_images)}")
    logger.info(f"{stage}: coverage = {coverage:.2f}")

    if coverage < 0.7:
        logger.warning("LOW coverage → reconstruction risk")
    elif coverage < 0.9:
        logger.info("acceptable coverage")
    else:
        logger.info("excellent coverage")

    logger.info(f"{stage}: SUCCESS")