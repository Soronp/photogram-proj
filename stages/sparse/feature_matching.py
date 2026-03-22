from pathlib import Path
import sqlite3


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
    except:
        return False


def run(paths, config, logger, tool_runner):
    stage = "feature_matching"
    logger.info(f"---- {stage.upper()} ----")

    db = paths.database

    if not db.exists():
        raise RuntimeError(f"{stage}: database missing")

    # 🔥 FORCE REMATCH (IMPORTANT)
    conn = sqlite3.connect(db)
    cur = conn.cursor()
    cur.execute("DELETE FROM matches;")
    cur.execute("DELETE FROM two_view_geometries;")
    conn.commit()
    conn.close()

    analysis = config.get("analysis_results", {})
    dataset = analysis.get("dataset", {})

    num_images = dataset.get("num_images", 0)

    # =========================================
    # 🔥 ALWAYS EXHAUSTIVE (FOR QUALITY)
    # =========================================
    matcher = "exhaustive"

    logger.info(f"{stage}: forcing exhaustive matching")

    # =========================================
    # 🔥 HIGH-RECALL MATCHING
    # =========================================
    max_ratio = 0.9         # 🔥 relaxed
    max_distance = 0.9      # 🔥 relaxed

    logger.info(
        f"{stage}: ratio={max_ratio}, distance={max_distance}"
    )

    def _cmd(use_gpu=True):
        return [
            "colmap", "exhaustive_matcher",
            "--database_path", str(db),

            "--SiftMatching.use_gpu", str(int(use_gpu)),
            "--SiftMatching.num_threads", "-1",

            # 🔥 CRITICAL CHANGES
            "--SiftMatching.max_ratio", str(max_ratio),
            "--SiftMatching.max_distance", str(max_distance),
            "--SiftMatching.cross_check", "0",   # 🔥 allow more matches
            "--SiftMatching.guided_matching", "1",
        ]

    try:
        tool_runner.run(_cmd(True), stage=stage + "_gpu")
    except:
        tool_runner.run(_cmd(False), stage=stage + "_cpu")

    if not _has_matches(db):
        raise RuntimeError(f"{stage}: matching failed")

    logger.info(f"{stage}: SUCCESS")