from pathlib import Path
import shutil

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _get_valid_images(folder: Path):
    return [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXT]


def _resolve_image_dir(paths, config):
    downsample_enabled = config.get("downsampling", {}).get("enabled", False)

    if downsample_enabled:
        ds_dir = paths.images_downsampled
        if ds_dir.exists():
            imgs = _get_valid_images(ds_dir)
            if len(imgs) > 0:
                return ds_dir, imgs

    orig_dir = paths.images
    if not orig_dir.exists():
        raise RuntimeError("mapper: original image directory missing")

    imgs = _get_valid_images(orig_dir)
    if len(imgs) == 0:
        raise RuntimeError("mapper: no valid images found")

    return orig_dir, imgs


# =====================================================
# VALIDATION
# =====================================================
def _validate_sparse_model(sparse_model: Path):
    required = ["cameras.bin", "images.bin", "points3D.bin"]
    missing = [f for f in required if not (sparse_model / f).exists()]
    if missing:
        raise RuntimeError(f"mapper: invalid sparse model → missing {missing}")


# =====================================================
# ENSURE BIN FORMAT (FOR GLOMAP SAFETY)
# =====================================================
def _ensure_binary_model(paths, tool_runner):
    sparse_model = paths.sparse_model

    if not (sparse_model / "cameras.bin").exists():
        tool_runner.run([
            "colmap", "model_converter",
            "--input_path", str(sparse_model),
            "--output_path", str(sparse_model),
            "--output_type", "BIN"
        ], stage="model_convert")


# =====================================================
# MAIN
# =====================================================
def run(paths, config, logger, tool_runner):
    stage = "mapper"
    logger.info(f"---- {stage.upper()} ----")

    image_dir, images = _resolve_image_dir(paths, config)
    logger.info(f"{stage}: using {len(images)} images from {image_dir}")

    database_path = paths.database
    if not database_path.exists():
        raise RuntimeError(f"{stage}: database missing → pipeline violated determinism")

    sparse_root = paths.sparse
    sparse_root.mkdir(parents=True, exist_ok=True)

    # 🔥 Clear previous sparse outputs ONLY
    for item in sparse_root.iterdir():
        if item.is_dir():
            shutil.rmtree(item)

    backend = config.get("pipeline", {}).get("sparse_backend", "colmap")
    retry = config.get("_meta", {}).get("retry_count", 0)
    analysis = config.get("analysis_results", {})
    matches = analysis.get("matches", {})
    connectivity = matches.get("connectivity", 0.3)

    logger.info(f"{stage}: backend={backend}, retry={retry}, connectivity={connectivity:.3f}")

    sparse_model = sparse_root / "0"

    # =====================================================
    # 🔥 GLOMAP BACKEND
    # =====================================================
    if backend == "glomap":

        logger.info(f"{stage}: running GLOMAP")

        try:
            cmd = [
                "glomap", "mapper",
                "--database_path", str(database_path),
                "--output_path", str(sparse_root),
            ]

            tool_runner.run(cmd, stage=stage + "_glomap")

            # 🔥 Ensure binary format
            _ensure_binary_model(paths, tool_runner)

            _validate_sparse_model(sparse_model)

            logger.info(f"{stage}: ✅ GLOMAP successful")

        except Exception as e:
            logger.warning(f"{stage}: GLOMAP failed → {e}")

            if config.get("sparse", {}).get("fallback_to_colmap", True):
                logger.info(f"{stage}: falling back to COLMAP")
                backend = "colmap"
            else:
                raise

    # =====================================================
    # 🔥 COLMAP BACKEND
    # =====================================================
    if backend == "colmap":

        logger.info(f"{stage}: running COLMAP")

        # -----------------------------
        # Adaptive parameters
        # -----------------------------
        init_inliers = 40
        abs_inliers = 30
        min_model_size = 15
        ba_global_iter = 50
        ba_local_iter = 25

        if connectivity < 0.2:
            init_inliers = 20
            abs_inliers = 15
            min_model_size = 5
        elif connectivity < 0.4:
            init_inliers = 30
            abs_inliers = 20
            min_model_size = 10

        if retry > 0:
            relax = 0.8 ** retry
            init_inliers = max(12, int(init_inliers * relax))
            abs_inliers = max(12, int(abs_inliers * relax))
            min_model_size = max(3, int(min_model_size * relax))
            ba_global_iter = int(ba_global_iter * (1 + 0.3 * retry))
            ba_local_iter = int(ba_local_iter * (1 + 0.2 * retry))

        logger.info(
            f"{stage}: inliers={init_inliers}/{abs_inliers}, "
            f"model_size={min_model_size}, BA={ba_global_iter}/{ba_local_iter}"
        )

        def _build_cmd(use_gpu=True):
            gpu_flag = "1" if use_gpu else "0"
            gpu_index = "0" if use_gpu else "-1"

            return [
                "colmap", "mapper",
                "--database_path", str(database_path),
                "--image_path", str(image_dir),
                "--output_path", str(sparse_root),
                "--Mapper.num_threads", "-1",
                "--Mapper.init_min_num_inliers", str(init_inliers),
                "--Mapper.abs_pose_min_num_inliers", str(abs_inliers),
                "--Mapper.min_model_size", str(min_model_size),
                "--Mapper.ba_global_max_num_iterations", str(ba_global_iter),
                "--Mapper.ba_local_max_num_iterations", str(ba_local_iter),
                "--Mapper.multiple_models", "0",
                "--Mapper.ba_use_gpu", gpu_flag,
                "--Mapper.ba_gpu_index", gpu_index,
            ]

        try:
            logger.info(f"{stage}: attempting GPU mapping...")
            tool_runner.run(_build_cmd(True), stage=stage + "_colmap_gpu")
            logger.info(f"{stage}: ✅ GPU mapping successful")
        except Exception as e:
            logger.warning(f"{stage}: GPU mapping failed: {e}")
            logger.info(f"{stage}: falling back to CPU mapping...")
            tool_runner.run(_build_cmd(False), stage=stage + "_colmap_cpu")
            logger.info(f"{stage}: ✅ CPU mapping successful")

        _validate_sparse_model(sparse_model)

    logger.info(f"{stage}: SUCCESS → deterministic sparse model ready")