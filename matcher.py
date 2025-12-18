#!/usr/bin/env python3
"""
matcher_dynamic.py

MARK-2 Exhaustive Matching (CPU/GPU-safe)
-----------------------------------------
- Updated for COLMAP latest versions
- Removes deprecated options
- Enforces stronger geometric verification
- Produces JSON coverage report
"""

import subprocess
import sqlite3
import json
from pathlib import Path
from math import comb
from utils.logger import get_logger
from utils.paths import ProjectPaths
from config_manager import create_runtime_config, validate_config

# ------------------------ UTILITIES ------------------------
def run_command(cmd, logger, label):
    cmd = [str(c) for c in cmd]
    logger.info(f"[RUN] {label}")
    logger.info(" ".join(cmd))
    try:
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        logger.info(result.stdout)
    except subprocess.CalledProcessError as e:
        logger.error(f"[FAIL] {label}")
        logger.error(e.stdout)
        raise RuntimeError(f"{label} failed") from e

def check_match_quality(db_path: Path, logger):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM images")
    image_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM matches")
    match_pair_count = cursor.fetchone()[0]

    expected_pairs = comb(image_count, 2)

    cursor.execute(
        "SELECT LENGTH(data) FROM matches WHERE LENGTH(data) > 150"
    )
    good_matches = cursor.fetchall()
    good_match_count = len(good_matches)

    coverage = (
        good_match_count / match_pair_count * 100
        if match_pair_count else 0.0
    )

    logger.info(
        f"Match statistics: {image_count} images, "
        f"{match_pair_count} matched pairs "
        f"({expected_pairs} possible), "
        f"{good_match_count} strong pairs "
        f"({coverage:.1f}%)"
    )

    conn.close()
    return {
        "images": image_count,
        "match_pairs": match_pair_count,
        "expected_pairs": expected_pairs,
        "good_matches": good_match_count,
        "coverage_percent": coverage,
    }

# ------------------------ MAIN MATCHER ------------------------
def run(project_root: Path, force: bool = False):
    paths = ProjectPaths(project_root)
    paths.ensure_all()
    logger = get_logger("matcher", project_root)

    config = create_runtime_config(project_root)
    if not validate_config(config, logger):
        logger.warning("Config validation failed — proceeding with defaults")

    db_path = paths.database / "database.db"
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    # ----------- FORCE CLEANUP -----------    
    if force:
        logger.info("Force enabled — clearing existing matches")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM matches")
        cursor.execute("DELETE FROM two_view_geometries")
        conn.commit()
        conn.close()

    # ----------- EXHAUSTIVE MATCHING (REMOVED DEPRECATED OPTIONS) -----------
    run_command(
        [
            "colmap", "exhaustive_matcher",
            "--database_path", str(db_path),

            # Modern COLMAP no longer needs guided_matching options
            "--SiftMatching.max_ratio", "0.8",
            "--SiftMatching.max_distance", "0.7",

            "--TwoViewGeometry.min_num_inliers", "25",
            "--TwoViewGeometry.max_error", "2.0",
        ],
        logger,
        "Exhaustive Matching"
    )

    # ----------- QUALITY CHECK -----------    
    stats = check_match_quality(db_path, logger)

    assessment = "good" if stats["coverage_percent"] >= 55.0 else "poor"

    report = {
        "strategy": "exhaustive",
        "database": str(db_path),
        "statistics": stats,
        "assessment": assessment,
    }

    report_path = paths.database / "matching_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Matching report saved to: {report_path}")

    if assessment == "poor":
        logger.warning(
            "Low match coverage detected — possible causes:\n"
            "- insufficient overlap\n"
            "- inconsistent exposure\n"
            "- remaining low-quality keypoints"
        )

    logger.info("Matching stage complete — ready for sparse reconstruction")

# ------------------------ CLI ------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="COLMAP exhaustive matcher (CPU/GPU-safe)"
    )
    parser.add_argument("--project", required=True, help="Project root")
    parser.add_argument("--force", action="store_true", help="Force rebuild matches")
    args = parser.parse_args()
    run(Path(args.project), args.force)
