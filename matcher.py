#!/usr/bin/env python3
"""
matcher_dynamic.py

MARK-2 Exhaustive Matching (Canonical)
-------------------------------------
- Runner-managed logger
- Resume-safe, force-aware
- Generates matching report
"""

import subprocess
import sqlite3
import json
from pathlib import Path
from math import comb

from utils.paths import ProjectPaths
from config_manager import create_runtime_config, validate_config


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def run_command(cmd, logger, label: str):
    logger.info(f"[matcher] RUN: {label}")
    logger.info(" ".join(map(str, cmd)))

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    logger.info(proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(f"{label} failed")


def match_stats(db: Path, logger):
    conn = sqlite3.connect(db)
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM images")
    imgs = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM matches")
    pairs = cur.fetchone()[0]

    cur.execute("SELECT LENGTH(data) FROM matches WHERE LENGTH(data) > 150")
    strong = len(cur.fetchall())

    conn.close()

    expected = comb(imgs, 2) if imgs >= 2 else 0
    coverage = (strong / pairs * 100) if pairs else 0.0

    logger.info(f"[matcher] Images: {imgs}, Pairs: {pairs}, Good matches: {strong}, Coverage: {coverage:.2f}%")
    return {
        "images": imgs,
        "match_pairs": pairs,
        "expected_pairs": expected,
        "good_matches": strong,
        "coverage_percent": coverage,
    }


# --------------------------------------------------
# Pipeline stage
# --------------------------------------------------
def run(project_root: Path, force: bool, logger):
    paths = ProjectPaths(project_root)
    paths.ensure_all()

    logger.info("[matcher] Starting exhaustive matching")

    # Load and validate config
    config = create_runtime_config(project_root, logger)
    validate_config(config, logger)

    db = paths.database / "database.db"
    if not db.exists():
        raise FileNotFoundError(db)

    if force:
        logger.info("[matcher] Force enabled — clearing matches")
        conn = sqlite3.connect(db)
        cur = conn.cursor()
        cur.execute("DELETE FROM matches")
        cur.execute("DELETE FROM two_view_geometries")
        conn.commit()
        conn.close()

    # Run COLMAP exhaustive matcher
    run_command(
        [
            "colmap", "exhaustive_matcher",
            "--database_path", db,
            "--SiftMatching.max_ratio", "0.8",
            "--SiftMatching.max_distance", "0.7",
            "--TwoViewGeometry.min_num_inliers", "25",
            "--TwoViewGeometry.max_error", "2.0",
        ],
        logger,
        "Exhaustive Matching"
    )

    # Collect statistics
    stats = match_stats(db, logger)
    assessment = "good" if stats["coverage_percent"] >= 55 else "poor"

    report = {
        "strategy": "exhaustive",
        "statistics": stats,
        "assessment": assessment,
    }

    report_path = paths.database / "matching_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    logger.info(f"[matcher] Matching report written: {report_path}")
    logger.info("[matcher] Matching complete")
