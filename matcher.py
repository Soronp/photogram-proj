#!/usr/bin/env python3
"""
matcher.py

MARK-2 Matching Stage
--------------------
- Uses exhaustive matching ONLY (deterministic, robust)
- Populates match tables in COLMAP database

Reads:
- database/database.db

Writes:
- database/database.db (matches)
- database/matching_report.json
- logs/matcher.log
"""

import subprocess
import sqlite3
import json
from pathlib import Path

from utils.logger import get_logger
from utils.paths import ProjectPaths
from config_manager import create_runtime_config, validate_config


# --------------------------------------------------
# Helper: Run subprocess commands
# --------------------------------------------------
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


# --------------------------------------------------
# Helper: Evaluate match quality
# --------------------------------------------------
def check_match_quality(db_path: Path, logger):
    """Simple match quality check without binary parsing."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Basic counts
    cursor.execute("SELECT COUNT(*) FROM images")
    image_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM matches")
    match_pair_count = cursor.fetchone()[0]
    
    expected_pairs = (image_count * (image_count - 1)) // 2

    logger.info("Match Statistics:")
    logger.info(f"  Images: {image_count}")
    logger.info(f"  Match pairs in database: {match_pair_count}")
    logger.info(f"  Expected pairs (exhaustive): {expected_pairs}")

    cursor.execute("SELECT LENGTH(data) as len FROM matches WHERE LENGTH(data) > 100")
    good_matches = cursor.fetchall()

    good_match_count = len(good_matches)
    coverage = (good_match_count / match_pair_count * 100) if match_pair_count > 0 else 0

    logger.info(f"  Match pairs with >100 bytes data: {good_match_count} ({coverage:.1f}%)")
    if coverage < 50:
        logger.warning(f"Low match quality - only {coverage:.1f}% of pairs have substantial data")

    conn.close()

    return {
        "images": image_count,
        "match_pairs": match_pair_count,
        "expected_pairs": expected_pairs,
        "good_matches": good_match_count,
        "coverage_percent": coverage
    }


# --------------------------------------------------
# Core: Run exhaustive matcher
# --------------------------------------------------
def run(project_root: Path, force: bool = False):
    paths = ProjectPaths(project_root)
    paths.ensure_all()

    logger = get_logger("matcher", project_root)

    # Load runtime config
    config = create_runtime_config(project_root)
    if not validate_config(config, logger):
        logger.warning("Config validation failed — proceeding with defaults")

    db_path = paths.database / "database.db"
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    logger.info("Starting exhaustive matching stage")
    logger.info(f"Database path: {db_path}")
    logger.info(f"Force rebuild: {force}")

    # Clear existing matches if force enabled
    if force:
        logger.info("Clearing existing matches...")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM matches")
        cursor.execute("DELETE FROM two_view_geometries")
        conn.commit()
        conn.close()

    # Run exhaustive matcher
    run_command(
        [
            "colmap",
            "exhaustive_matcher",
            "--database_path", str(db_path),
        ],
        logger,
        "Exhaustive Matching"
    )

    # Evaluate match quality
    stats = check_match_quality(db_path, logger)

    # Generate JSON report
    report = {
        "strategy": "exhaustive",
        "database": str(db_path),
        "statistics": stats,
        "assessment": "good" if stats["coverage_percent"] > 50 else "poor"
    }

    report_path = paths.database / "matching_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Matching report saved to: {report_path}")

    if stats["coverage_percent"] < 50:
        logger.warning("Poor match coverage detected. Possible issues:")
        logger.warning("  - Images have insufficient overlap")
        logger.warning("  - Feature extraction may have failed")
        logger.warning("  - Image dimensions may be inconsistent")

    logger.info("Matching completed successfully")


# --------------------------------------------------
# CLI wrapper
# --------------------------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(description="MARK-2 COLMAP Exhaustive Matcher")
    parser.add_argument("--project", required=True, help="Project root directory")
    parser.add_argument("--force", action="store_true", help="Force rebuild matches")
    args = parser.parse_args()

    run(Path(args.project), args.force)


if __name__ == "__main__":
    main()
