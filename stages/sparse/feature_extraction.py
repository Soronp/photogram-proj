from pathlib import Path
import sqlite3
import subprocess
import sys

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _get_valid_images(folder: Path):
    return [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXT]


def _resolve_image_dir(paths, config, logger):
    """
    Decide whether to use downsampled images or originals.
    Must match mapper logic.
    """
    downsample_enabled = config.get("downsampling", {}).get("enabled", False)
    if downsample_enabled:
        ds_dir = paths.images_downsampled
        if ds_dir.exists():
            imgs = _get_valid_images(ds_dir)
            if imgs:
                logger.info(f"feature_extraction: using DOWN SAMPLED images ({len(imgs)})")
                print(f"[INFO] Using downsampled images ({len(imgs)})")
                return ds_dir, imgs

    # fallback
    orig_dir = paths.images
    if not orig_dir.exists():
        raise RuntimeError("feature_extraction: original image directory missing")

    imgs = _get_valid_images(orig_dir)
    if not imgs:
        raise RuntimeError("feature_extraction: no valid images found")

    logger.info(f"feature_extraction: using ORIGINAL images ({len(imgs)})")
    print(f"[INFO] Using original images ({len(imgs)})")
    return orig_dir, imgs


def _database_has_features(database_path: Path):
    if not database_path.exists():
        return False
    try:
        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM images;")
        num_images = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM keypoints;")
        num_keypoints = cursor.fetchone()[0]
        conn.close()
        return num_images > 0 and num_keypoints > 0
    except Exception:
        return False


def _run_command(cmd, stage, logger):
    """Run subprocess command and stream output to stdout and logger."""
    logger.info(f"{stage}: running → {' '.join(cmd)}")
    print(f"[INFO] Running {stage}: {' '.join(cmd)}")

    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    for line in process.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"{stage} failed with exit code {process.returncode}")


def run(paths, config, logger, tool_runner):
    stage = "feature_extraction"
    logger.info(f"---- {stage.upper()} ----")
    print(f"[INFO] ---- {stage.upper()} ----")

    image_dir, images = _resolve_image_dir(paths, config, logger)
    database_path = paths.database

    # Skip if database already has features
    if _database_has_features(database_path):
        logger.info(f"{stage}: valid database found → SKIPPING extraction")
        print(f"[INFO] Database already has features, skipping extraction")
        return
    elif database_path.exists():
        logger.warning(f"{stage}: removing invalid/empty database")
        print(f"[WARN] Removing invalid/empty database")
        database_path.unlink()

    # -----------------------------
    # Adaptive feature settings
    # -----------------------------
    sift_cfg = config.get("sift", {})
    backend = config.get("sparse", {}).get("backend", "colmap")
    retry = config.get("_meta", {}).get("retry_count", 0)
    num_images = config.get("analysis_results", {}).get("dataset", {}).get("num_images", len(images))

    # High-first strategy
    if num_images < 50:
        max_features, max_img_size = 20000, 3200
    elif num_images < 150:
        max_features, max_img_size = 18000, 3200
    elif num_images < 400:
        max_features, max_img_size = 16000, 2800
    else:
        max_features, max_img_size = 12000, 2200

    # Retry scaling
    if retry > 0:
        scale = 0.85 ** retry
        max_features = int(max_features * scale)
        max_img_size = int(max_img_size * scale)
        logger.info(f"{stage}: retry adjustment {retry} → scaling compute")
        print(f"[INFO] Retry adjustment {retry}, max_features → {max_features}, max_img_size → {max_img_size}")

    max_features = max(4000, min(max_features, 20000))
    max_img_size = max(1200, min(max_img_size, 3200))

    # Backend tuning
    if backend == "glomap":
        peak_threshold, edge_threshold = 0.006, 10
        logger.info(f"{stage}: GLOMAP mode")
        print(f"[INFO] GLOMAP backend mode")
    else:
        peak_threshold, edge_threshold = 0.004, 12
        logger.info(f"{stage}: COLMAP mode")
        print(f"[INFO] COLMAP backend mode")

    # GPU-first
    use_gpu = sift_cfg.get("use_gpu", True)
    gpu_flag = 1 if use_gpu else 0
    logger.info(f"{stage}: features={max_features}, img_size={max_img_size}, GPU={use_gpu}")
    print(f"[INFO] Feature extraction: features={max_features}, max_img_size={max_img_size}, GPU={use_gpu}")

    # -----------------------------
    # Feature extraction CLI
    # -----------------------------
    cmd = [
        "colmap", "feature_extractor",
        "--database_path", str(database_path),
        "--image_path", str(image_dir),
        "--SiftExtraction.use_gpu", str(gpu_flag),
        "--SiftExtraction.gpu_index", "-1",
        "--SiftExtraction.max_num_features", str(max_features),
        "--SiftExtraction.max_image_size", str(max_img_size),
        "--SiftExtraction.num_threads", "-1",
        "--SiftExtraction.peak_threshold", str(peak_threshold),
        "--SiftExtraction.edge_threshold", str(edge_threshold),
        "--SiftExtraction.first_octave", "-1",
        "--SiftExtraction.num_octaves", "4",
        "--SiftExtraction.octave_resolution", "3",
        "--SiftExtraction.upright", "0",
    ]

    try:
        tool_runner.run(cmd, stage=stage)
    except RuntimeError as e:
        if use_gpu:
            logger.warning(f"{stage}: GPU failed, falling back to CPU → {e}")
            print(f"[WARN] GPU failed, falling back to CPU")
            cmd[cmd.index(str(gpu_flag))] = "0"  # force CPU
            tool_runner.run(cmd, stage=stage)
        else:
            raise

    # -----------------------------
    # Validate extraction
    # -----------------------------
    if not _database_has_features(database_path):
        raise RuntimeError(f"{stage}: extraction failed (no images/features in database)")
    logger.info(f"{stage}: SUCCESS")
    print(f"[INFO] Feature extraction SUCCESS")

    # -----------------------------
    # Feature matching (GPU first)
    # -----------------------------
    stage_match = "feature_matching"
    logger.info(f"---- {stage_match.upper()} ----")
    print(f"[INFO] ---- {stage_match.upper()} ----")

    cmd_match = [
        "colmap", "exhaustive_matcher",
        "--database_path", str(database_path),
        "--SiftMatching.use_gpu", str(gpu_flag),
        "--SiftMatching.num_threads", "-1",
    ]

    try:
        tool_runner.run(cmd_match, stage=stage_match)
    except RuntimeError as e:
        if use_gpu:
            logger.warning(f"{stage_match}: GPU matching failed, falling back to CPU → {e}")
            print(f"[WARN] GPU matching failed, falling back to CPU")
            cmd_match[cmd_match.index(str(gpu_flag))] = "0"
            tool_runner.run(cmd_match, stage=stage_match)
        else:
            raise

    # Validate matches
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM matches;")
    num_matches = cursor.fetchone()[0]
    conn.close()
    logger.info(f"{stage_match}: {num_matches} matches found")
    print(f"[INFO] Feature matching: {num_matches} matches found")