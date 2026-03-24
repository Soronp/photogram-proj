from pathlib import Path
import shutil


def _find_best_sparse_model(sparse_root: Path):
    models = [p for p in sparse_root.iterdir() if p.is_dir()]

    if not models:
        raise RuntimeError("image_undistorter: no sparse models found")

    def score(model):
        pts = model / "points3D.bin"
        if not pts.exists():
            return 0
        return pts.stat().st_size  # 🔥 proxy for reconstruction quality

    best = max(models, key=score)

    if score(best) == 0:
        raise RuntimeError("image_undistorter: no valid sparse model (points3D missing)")

    return best


def run(paths, config, logger, tool_runner):
    stage = "image_undistorter"
    logger.info(f"---- {stage.upper()} ----")

    sparse_root = paths.sparse
    image_dir = paths.images
    dense_dir = paths.dense

    # =====================================================
    # 🔥 VALIDATION
    # =====================================================
    if not sparse_root.exists():
        raise RuntimeError(f"{stage}: sparse folder missing")

    if not image_dir.exists():
        raise RuntimeError(f"{stage}: image directory missing")

    dense_dir.mkdir(parents=True, exist_ok=True)

    # =====================================================
    # 🔥 SELECT BEST SPARSE MODEL
    # =====================================================
    sparse_model = _find_best_sparse_model(sparse_root)
    logger.info(f"{stage}: using sparse model → {sparse_model.name}")

    # =====================================================
    # 🔥 ANALYSIS INPUT
    # =====================================================
    analysis = config.get("analysis_results", {})
    dataset = analysis.get("dataset", {})

    num_images = dataset.get("num_images", 0)
    avg_resolution = dataset.get("avg_resolution", 2000)
    retry = config.get("_meta", {}).get("retry_count", 0)

    # =====================================================
    # 🔥 RESOLUTION STRATEGY (MORE AGGRESSIVE)
    # =====================================================
    # 🔥 KEY CHANGE: prioritize detail, not safety
    if retry == 0:
        max_image_size = 3000
        logger.info(f"{stage}: HIGH DETAIL mode")

    elif retry == 1:
        max_image_size = 3600
        logger.info(f"{stage}: BOOST DETAIL mode")

    else:
        max_image_size = 4000
        logger.info(f"{stage}: MAX DETAIL mode")

    # Adaptive clamp based on dataset
    if avg_resolution < 1800:
        max_image_size = min(max_image_size, 2600)

    max_image_size = min(max_image_size, 4000)

    logger.info(
        f"{stage}: images={num_images}, "
        f"avg_res={avg_resolution}, "
        f"max_image_size={max_image_size}"
    )

    # =====================================================
    # 🔥 CLEAN PREVIOUS DENSE OUTPUT (SAFE)
    # =====================================================
    dense_images_dir = dense_dir / "images"
    dense_sparse_dir = dense_dir / "sparse"

    if dense_images_dir.exists():
        logger.warning(f"{stage}: clearing previous dense/images")
        shutil.rmtree(dense_images_dir)

    if dense_sparse_dir.exists():
        logger.warning(f"{stage}: clearing previous dense/sparse")
        shutil.rmtree(dense_sparse_dir)

    # =====================================================
    # 🔥 COMMAND
    # =====================================================
    cmd = [
        "colmap",
        "image_undistorter",
        "--image_path", str(image_dir),
        "--input_path", str(sparse_model),
        "--output_path", str(dense_dir),
        "--output_type", "COLMAP",

        # CRITICAL: drives ALL downstream density
        "--max_image_size", str(max_image_size),
    ]

    # =====================================================
    # 🔥 EXECUTION
    # =====================================================
    tool_runner.run(cmd, stage=stage)

    # =====================================================
    # 🔥 VALIDATION
    # =====================================================
    if not dense_images_dir.exists():
        raise RuntimeError(f"{stage}: undistortion failed (no images output)")

    out_images = list(dense_images_dir.glob("*"))
    in_images = list(image_dir.glob("*"))

    num_out = len(out_images)
    num_in = len(in_images)

    coverage = num_out / max(num_in, 1)

    logger.info(f"{stage}: output images = {num_out}")
    logger.info(f"{stage}: coverage = {coverage:.2f}")

    if coverage < 0.7:
        logger.warning(f"{stage}: LOW undistortion coverage → downstream holes likely")
    elif coverage < 0.9:
        logger.info(f"{stage}: acceptable coverage")
    else:
        logger.info(f"{stage}: excellent coverage")

    # =====================================================
    # 🔥 FINAL STRUCTURE CHECK (IMPORTANT)
    # =====================================================
    if not dense_sparse_dir.exists():
        raise RuntimeError(f"{stage}: missing dense/sparse output (invalid workspace)")

    logger.info(f"{stage}: SUCCESS")