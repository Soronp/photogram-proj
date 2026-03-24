from pathlib import Path
import sqlite3

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# =====================================================
# HELPERS
# =====================================================
def _get_valid_images(folder: Path):
    return [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXT
    ]


def _has_features(db: Path):
    if not db.exists():
        return False

    try:
        conn = sqlite3.connect(db)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM keypoints;")
        count = cur.fetchone()[0]
        conn.close()
        return count > 0
    except Exception:
        return False


def _resolve_image_dir(paths, config, logger):
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
    # CLEAN DATABASE
    # =====================================================
    if database_path.exists():
        logger.warning(f"{stage}: removing existing database")
        database_path.unlink()

    # =====================================================
    # SIFT CONFIG
    # =====================================================
    sift_cfg = config.get("sift", {})
    use_gpu = sift_cfg.get("use_gpu", True)

    max_features = sift_cfg.get("max_num_features", 18000)
    max_img_size = sift_cfg.get("max_image_size", 2800)

    peak_threshold = sift_cfg.get("peak_threshold", 0.003)
    edge_threshold = sift_cfg.get("edge_threshold", 10)
    first_octave = sift_cfg.get("first_octave", -1)

    # =====================================================
    # CAMERA MODEL (🔥 CRITICAL FIX FOR PIPELINE C)
    # =====================================================
    camera_model = config.get("pipeline", {}).get("camera_model", "OPENCV")

    logger.info(
        f"{stage}: config → features={max_features}, "
        f"img_size={max_img_size}, camera_model={camera_model}"
    )

    # =====================================================
    # COMMAND BUILDER
    # =====================================================
    def _build_cmd(use_gpu_flag):
        return [
            "colmap", "feature_extractor",

            "--database_path", str(database_path),
            "--image_path", str(image_dir),

            # GPU / CPU
            "--SiftExtraction.use_gpu", str(int(use_gpu_flag)),
            "--SiftExtraction.gpu_index", "0" if use_gpu_flag else "-1",

            # SIFT params
            "--SiftExtraction.max_num_features", str(max_features),
            "--SiftExtraction.max_image_size", str(max_img_size),
            "--SiftExtraction.peak_threshold", str(peak_threshold),
            "--SiftExtraction.edge_threshold", str(edge_threshold),
            "--SiftExtraction.first_octave", str(first_octave),

            # 🔥 CAMERA MODEL FIX
            "--ImageReader.camera_model", camera_model,

            "--SiftExtraction.num_threads", "-1",
        ]

    # =====================================================
    # EXECUTION (GPU → CPU fallback)
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
    # VALIDATION
    # =====================================================
    if not _has_features(database_path):
        raise RuntimeError(f"{stage}: extraction failed")

    conn = sqlite3.connect(database_path)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM keypoints;")
    num_keypoints = cur.fetchone()[0]
    conn.close()

    logger.info(f"{stage}: total keypoints = {num_keypoints}")
    logger.info(f"{stage}: SUCCESS")