from pathlib import Path


def _find_sparse_model(sparse_root: Path):
    models = [p for p in sparse_root.iterdir() if p.is_dir() and any(p.iterdir())]
    if not models:
        raise RuntimeError("No valid sparse models found")

    # 🔥 pick largest model (most files)
    return sorted(models, key=lambda p: len(list(p.iterdir())), reverse=True)[0]


def run(paths, config, logger, tool_runner):
    stage = "image_undistorter"
    logger.info(f"---- {stage.upper()} ----")

    sparse_root = paths.sparse

    if not sparse_root.exists():
        raise RuntimeError(f"{stage}: sparse folder missing")

    sparse_model = _find_sparse_model(sparse_root)
    logger.info(f"{stage}: using sparse model → {sparse_model.name}")

    image_dir = paths.images
    if not image_dir.exists():
        raise RuntimeError(f"{stage}: image directory missing")

    dense_dir = paths.dense
    dense_dir.mkdir(parents=True, exist_ok=True)

    # =====================================================
    # 🔥 ANALYSIS INPUT
    # =====================================================
    analysis = config.get("analysis_results", {})
    dataset = analysis.get("dataset", {})

    num_images = dataset.get("num_images", 0)
    avg_resolution = dataset.get("avg_resolution", 2000)
    retry = config.get("_meta", {}).get("retry_count", 0)

    # =====================================================
    # 🔥 ADAPTIVE MAX IMAGE SIZE (CRITICAL FOR DENSITY)
    # =====================================================
    if retry == 0:
        if avg_resolution >= 3000:
            max_image_size = 3000
        elif avg_resolution >= 2000:
            max_image_size = 2600
        else:
            max_image_size = 2200
        logger.info(f"{stage}: HIGH-RES mode")

    elif retry == 1:
        max_image_size = 3000
        logger.info(f"{stage}: BOOST mode")

    else:
        max_image_size = 3200
        logger.info(f"{stage}: MAX QUALITY mode")

    # Hard cap
    max_image_size = min(max_image_size, 3200)

    logger.info(f"{stage}: max_image_size={max_image_size}")

    # =====================================================
    # 🔥 FORCE REBUILD (IMPORTANT FOR REFINEMENT LOOP)
    # =====================================================
    dense_images_dir = dense_dir / "images"

    if dense_images_dir.exists():
        logger.warning(f"{stage}: removing previous undistorted images")
        for f in dense_images_dir.glob("*"):
            f.unlink()
        dense_images_dir.rmdir()

    # =====================================================
    # 🔥 BUILD COMMAND
    # =====================================================
    cmd = [
        "colmap",
        "image_undistorter",
        "--image_path", str(image_dir),
        "--input_path", str(sparse_model),
        "--output_path", str(dense_dir),
        "--output_type", "COLMAP",

        # 🔥 CORE DENSITY CONTROL
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
        raise RuntimeError(f"{stage}: undistortion failed (no images)")

    num_out = len(list(dense_images_dir.glob("*")))
    num_in = len(list(image_dir.glob("*")))

    coverage = num_out / max(num_in, 1)

    logger.info(f"{stage}: output images = {num_out}")
    logger.info(f"{stage}: coverage = {coverage:.2f}")

    if coverage < 0.7:
        logger.warning(f"{stage}: LOW undistortion coverage")
    else:
        logger.info(f"{stage}: GOOD undistortion coverage")

    logger.info(f"{stage}: SUCCESS")