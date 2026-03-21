from pathlib import Path
import sqlite3

def _has_features(database_path: Path):
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

def _has_matches(database_path: Path):
    if not database_path.exists():
        return False
    try:
        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM matches;")
        num_matches = cursor.fetchone()[0]
        conn.close()
        return num_matches > 0
    except Exception:
        return False

def run(paths, config, logger, tool_runner):
    stage = "feature_matching"
    logger.info(f"---- {stage.upper()} ----")

    database_path = paths.database

    if not database_path.exists():
        raise RuntimeError(f"{stage}: database missing")

    # =====================================================
    # ✅ PRE-VALIDATION
    # =====================================================
    if not _has_features(database_path):
        raise RuntimeError(f"{stage}: database has no features/images")

    backend = config.get("sparse", {}).get("backend", "colmap")
    retry = config.get("_meta", {}).get("retry_count", 0)
    num_images = config.get("analysis_results", {}).get("dataset", {}).get("num_images", 0)

    logger.info(f"{stage}: backend={backend}, retry={retry}, images={num_images}")
    logger.info(f"{stage}: database_path={database_path}")

    # =====================================================
    # 🔥 BASELINE MATCHING PARAMS
    # =====================================================
    if backend == "glomap":
        max_ratio = 0.75
        max_distance = 0.7
        guided = 1
        logger.info(f"{stage}: GLOMAP strict matching")
    else:
        max_ratio = 0.85
        max_distance = 0.8
        guided = 1
        logger.info(f"{stage}: COLMAP adaptive matching")

    # =====================================================
    # 🔥 RETRY → RELAX MATCHING
    # =====================================================
    if retry > 0:
        relax_factor = 1 + (0.1 * retry)
        max_ratio = min(0.95, max_ratio * relax_factor)
        max_distance = min(1.0, max_distance * relax_factor)
        logger.info(f"{stage}: retry={retry} → relaxing matching")

    # =====================================================
    # 🔥 SELECT MATCHER TYPE
    # =====================================================
    if backend == "glomap":
        matcher = "exhaustive"
    else:
        if num_images < 80:
            matcher = "exhaustive"
        elif num_images < 300:
            matcher = "sequential"
        else:
            matcher = "vocab_tree"

    logger.info(f"{stage}: matcher={matcher}")

    # =====================================================
    # 🔥 RUN MATCHING GPU-FIRST
    # =====================================================
    def _build_cmd(use_gpu=True):
        gpu_flag = "1" if use_gpu else "0"

        if matcher == "exhaustive":
            return [
                "colmap", "exhaustive_matcher",
                "--database_path", str(database_path),
                "--SiftMatching.use_gpu", gpu_flag,
                "--SiftMatching.num_threads", "-1",
                "--SiftMatching.max_ratio", str(max_ratio),
                "--SiftMatching.max_distance", str(max_distance),
                "--SiftMatching.cross_check", "1",
                "--SiftMatching.guided_matching", str(guided),
            ]
        elif matcher == "sequential":
            return [
                "colmap", "sequential_matcher",
                "--database_path", str(database_path),
                "--SiftMatching.use_gpu", gpu_flag,
                "--SiftMatching.num_threads", "-1",
                "--SiftMatching.guided_matching", "1",
                "--SequentialMatching.overlap", "5",
            ]
        elif matcher == "vocab_tree":
            return [
                "colmap", "vocab_tree_matcher",
                "--database_path", str(database_path),
                "--SiftMatching.use_gpu", gpu_flag,
                "--SiftMatching.num_threads", "-1",
            ]
        else:
            raise ValueError(f"{stage}: unknown matcher type")

    # Attempt GPU first
    try:
        logger.info(f"{stage}: attempting GPU matching...")
        tool_runner.run(_build_cmd(use_gpu=True), stage=stage)
        if not _has_matches(database_path):
            raise RuntimeError("GPU matching produced 0 matches")
        logger.info(f"{stage}: ✅ GPU matching successful")
    except Exception as e:
        logger.warning(f"{stage}: GPU matching failed: {e}")
        logger.info(f"{stage}: falling back to CPU matching...")
        tool_runner.run(_build_cmd(use_gpu=False), stage=stage)
        if not _has_matches(database_path):
            raise RuntimeError(f"{stage}: CPU fallback also failed (0 matches)")
        logger.info(f"{stage}: ✅ CPU matching successful")