from pathlib import Path
import sqlite3

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


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
    except:
        return False


def run(paths, config, logger, tool_runner):
    stage = "feature_extraction"
    logger.info(f"---- {stage.upper()} ----")

    image_dir = paths.images
    database_path = paths.database

    images = _get_valid_images(image_dir)
    if not images:
        raise RuntimeError(f"{stage}: no images found")

    # 🔥 FORCE REBUILD (IMPORTANT FOR QUALITY)
    if database_path.exists():
        database_path.unlink()

    # =========================================
    # 🔥 HIGH-COVERAGE SIFT CONFIG
    # =========================================
    use_gpu = config.get("sift", {}).get("use_gpu", True)

    max_features = 24000        # 🔥 increased
    max_img_size = 2800         # 🔥 higher detail

    logger.info(
        f"{stage}: HIGH COVERAGE MODE | "
        f"features={max_features}, img_size={max_img_size}"
    )

    def _cmd(use_gpu_flag):
        return [
            "colmap", "feature_extractor",
            "--database_path", str(database_path),
            "--image_path", str(image_dir),

            "--SiftExtraction.use_gpu", str(int(use_gpu_flag)),
            "--SiftExtraction.gpu_index", "0" if use_gpu_flag else "-1",

            "--SiftExtraction.max_num_features", str(max_features),
            "--SiftExtraction.max_image_size", str(max_img_size),

            # 🔥 CRITICAL CHANGES
            "--SiftExtraction.peak_threshold", "0.002",   # more features
            "--SiftExtraction.edge_threshold", "8",       # better edges
            "--SiftExtraction.first_octave", "-1",        # finer detail

            "--SiftExtraction.num_threads", "-1",
        ]

    try:
        tool_runner.run(_cmd(True), stage=stage + "_gpu")
    except:
        tool_runner.run(_cmd(False), stage=stage + "_cpu")

    if not _has_features(database_path):
        raise RuntimeError(f"{stage}: extraction failed")

    logger.info(f"{stage}: SUCCESS")