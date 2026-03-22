from pathlib import Path
import sqlite3

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# =====================================================
# HELPERS
# =====================================================
def _get_valid_images(folder: Path):
    return [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXT]


def _has_features(db: Path):
    if not db.exists():
        return False
    try:
        conn = sqlite3.connect(db)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM keypoints;")
        n = cur.fetchone()[0]
        conn.close()
        return n > 0
    except Exception:
        return False


def _resolve_image_dir(paths, config, logger):
    """
    Prefer downsampled images if enabled and valid.
    """
    downsample_enabled = config.get("downsampling", {}).get("enabled", False)

    if downsample_enabled and paths.images_downsampled.exists():
        imgs = _get_valid_images(paths.images_downsampled)
        if imgs:
            logger.info(f"feature_extraction: using downsampled ({len(imgs)})")
            return paths.images_downsampled, imgs

    imgs = _get_valid_images(paths.images)
    if not imgs:
        raise RuntimeError("feature_extraction: no valid images found")

    logger.info(f"feature_extraction: using original ({len(imgs)})")
    return paths.images, imgs


# =====================================================
# MAIN
# =====================================================
def run(paths, config, logger, tool_runner):
    stage = "feature_extraction"
    logger.info(f"---- {stage.upper()} ----")

    image_dir, images = _resolve_image_dir(paths, config, logger)
    database_path = paths.database

    # =====================================================
    # 🔥 FORCE CLEAN REBUILD (IMPORTANT)
    # =====================================================
    if database_path.exists():
        logger.warning(f"{stage}: removing existing database")
        database_path.unlink()

    # =====================================================
    # 🔥 BALANCED HIGH-COVERAGE CONFIG
    # =====================================================
    sift_cfg = config.get("sift", {})
    use_gpu = sift_cfg.get("use_gpu", True)

    # 🔥 KEY BALANCE (not overkill)
    max_features = 18000
    max_img_size = 2800

    # 🔥 OPTIONAL BOOST (controlled)
    peak_threshold = 0.003     # lower → more features, but not noisy
    edge_threshold = 10        # keep stable edges
    first_octave = -1          # capture fine details

    logger.info(
        f"{stage}: BALANCED COVERAGE MODE | "
        f"features={max_features}, img_size={max_img_size}, "
        f"peak={peak_threshold}"
    )

    # =====================================================
    # 🔥 COMMAND BUILDER
    # =====================================================
    def _build_cmd(use_gpu_flag):
        return [
            "colmap", "feature_extractor",
            "--database_path", str(database_path),
            "--image_path", str(image_dir),

            "--SiftExtraction.use_gpu", str(int(use_gpu_flag)),
            "--SiftExtraction.gpu_index", "0" if use_gpu_flag else "-1",

            "--SiftExtraction.max_num_features", str(max_features),
            "--SiftExtraction.max_image_size", str(max_img_size),

            # 🔥 CRITICAL TUNING
            "--SiftExtraction.peak_threshold", str(peak_threshold),
            "--SiftExtraction.edge_threshold", str(edge_threshold),
            "--SiftExtraction.first_octave", str(first_octave),

            "--SiftExtraction.num_threads", "-1",
        ]

    # =====================================================
    # 🔥 EXECUTION (GPU → CPU)
    # =====================================================
    try:
        if use_gpu:
            logger.info(f"{stage}: running on GPU")
            tool_runner.run(_build_cmd(True), stage=stage + "_gpu")
        else:
            raise RuntimeError("GPU disabled")

    except Exception as e:
        logger.warning(f"{stage}: GPU failed → {e}")
        logger.info(f"{stage}: falling back to CPU")
        tool_runner.run(_build_cmd(False), stage=stage + "_cpu")

    # =====================================================
    # 🔥 VALIDATION
    # =====================================================
    if not _has_features(database_path):
        raise RuntimeError(f"{stage}: extraction failed")

    # Diagnostics
    conn = sqlite3.connect(database_path)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM keypoints;")
    num_keypoints = cur.fetchone()[0]
    conn.close()

    logger.info(f"{stage}: total keypoints = {num_keypoints}")
    logger.info(f"{stage}: SUCCESS")