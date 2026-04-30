from pathlib import Path
import sqlite3


# =====================================================
# VALIDATION HELPERS
# =====================================================
def _has_matches(db: Path):
    if not db.exists():
        return False
    try:
        conn = sqlite3.connect(db)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM matches;")
        n = cur.fetchone()[0]
        conn.close()
        return n > 0
    except Exception:
        return False


def _count_matches(db: Path):
    try:
        conn = sqlite3.connect(db)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM matches;")
        n = cur.fetchone()[0]
        conn.close()
        return n
    except Exception:
        return 0


# =====================================================
# MAIN MATCHING
# =====================================================
def run(paths, config, logger, tool_runner):

    stage = "feature_matching"
    logger.info(f"---- {stage.upper()} ----")

    db = paths.database

    if not db.exists():
        raise RuntimeError(f"{stage}: database missing")

    # =====================================================
    # FORCE CLEAN MATCH STATE
    # =====================================================
    logger.info(f"{stage}: clearing previous matches")

    conn = sqlite3.connect(db)
    cur = conn.cursor()
    cur.execute("DELETE FROM matches;")
    cur.execute("DELETE FROM two_view_geometries;")
    conn.commit()
    conn.close()

    # =====================================================
    # ANALYSIS INPUT
    # =====================================================
    analysis = config.get("analysis_results", {})
    dataset = analysis.get("dataset", {})

    num_images = dataset.get("num_images", 0)
    retry = config.get("_meta", {}).get("retry_count", 0)

    logger.info(f"{stage}: images={num_images}, retry={retry}")

    # =====================================================
    # MATCHING STRATEGY
    # =====================================================
    # Always exhaustive for object reconstruction
    matcher = "exhaustive"
    logger.info(f"{stage}: matcher=exhaustive (forced)")

    # =====================================================
    # BASELINE (GEOMETRY-FIRST)
    # =====================================================
    max_ratio = 0.75
    max_distance = 0.7
    cross_check = 1
    guided_matching = 1

    # =====================================================
    # RETRY ADAPTATION (CONTROLLED)
    # =====================================================
    if retry > 0:
        logger.info(f"{stage}: retry level {retry} → relaxing constraints")

        max_ratio = min(0.85, max_ratio + 0.03 * retry)
        max_distance = min(0.8, max_distance + 0.03 * retry)

        # Only relax cross-check if needed
        if retry >= 2:
            cross_check = 0

    # =====================================================
    # SMALL DATASET BOOST (IMPORTANT FOR YOUR CASE)
    # =====================================================
    if num_images <= 50:
        logger.info(f"{stage}: small dataset → boosting recall slightly")
        max_ratio = min(max_ratio + 0.02, 0.85)

    logger.info(
        f"{stage}: ratio={max_ratio:.3f}, "
        f"distance={max_distance:.3f}, "
        f"cross_check={cross_check}"
    )

    # =====================================================
    # COMMAND BUILDER
    # =====================================================
    def _cmd(use_gpu=True):
        return [
            "colmap", "exhaustive_matcher",
            "--database_path", str(db),
            "--SiftMatching.use_gpu", str(int(use_gpu)),
            "--SiftMatching.num_threads", "-1",
            "--SiftMatching.max_ratio", str(max_ratio),
            "--SiftMatching.max_distance", str(max_distance),
            "--SiftMatching.cross_check", str(cross_check),
            "--SiftMatching.guided_matching", str(guided_matching),
            "--ExhaustiveMatching.block_size", "50",
        ]

    # =====================================================
    # EXECUTION (GPU → CPU)
    # =====================================================
    try:
        logger.info(f"{stage}: running GPU matching...")
        tool_runner.run(_cmd(True), stage=stage + "_gpu")
    except Exception as e:
        logger.warning(f"{stage}: GPU failed → {e}")
        logger.info(f"{stage}: falling back to CPU...")
        tool_runner.run(_cmd(False), stage=stage + "_cpu")

    # =====================================================
    # VALIDATION
    # =====================================================
    if not _has_matches(db):
        raise RuntimeError(f"{stage}: matching failed")

    total_matches = _count_matches(db)
    logger.info(f"{stage}: total matches = {total_matches}")

    # =====================================================
    # QUALITY DIAGNOSTICS
    # =====================================================
    if total_matches < 2000:
        logger.warning(f"{stage}: VERY LOW matches → reconstruction likely incomplete")

    elif total_matches < 15000:
        logger.info(f"{stage}: moderate match density")

    else:
        logger.info(f"{stage}: strong match density")

    logger.info(f"{stage}: SUCCESS")