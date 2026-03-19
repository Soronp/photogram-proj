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

    # -----------------------------
    # Resolve sparse model
    # -----------------------------
    sparse_root = paths.sparse

    if not sparse_root.exists():
        raise RuntimeError(f"{stage}: sparse folder missing")

    sparse_model = _find_sparse_model(sparse_root)
    logger.info(f"{stage}: using sparse model → {sparse_model.name}")

    # 🔥 ALWAYS use canonical image dir
    image_dir = paths.images

    if not image_dir.exists():
        raise RuntimeError(f"{stage}: image directory missing")

    # -----------------------------
    # Output
    # -----------------------------
    dense_dir = paths.dense
    dense_dir.mkdir(parents=True, exist_ok=True)

    if (dense_dir / "images").exists():
        logger.warning(f"{stage}: already done, skipping")
        return

    # -----------------------------
    # Config (SAFE DEFAULT)
    # -----------------------------
    max_image_size = config.get("dense", {}).get("max_image_size", 1600)

    cmd = [
        "colmap",
        "image_undistorter",
        "--image_path", str(image_dir),
        "--input_path", str(sparse_model),
        "--output_path", str(dense_dir),
        "--output_type", "COLMAP",
        "--max_image_size", str(max_image_size),
    ]

    tool_runner.run(cmd, stage=stage)

    logger.info(f"{stage}: DONE")