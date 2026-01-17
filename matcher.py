#!/usr/bin/env python3
"""
matcher_dynamic.py

MARK-2 Exhaustive Matching Stage
--------------------------------
Responsibilities:
- Resume-safe COLMAP exhaustive matching
- Force-aware database cleanup
- Matching quality assessment + report generation
"""

import subprocess
import sqlite3
import json
from pathlib import Path
from math import comb

from utils.paths import ProjectPaths
from config_manager import create_runtime_config, validate_config


# --------------------------------------------------
# Command Runner
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

    if proc.stdout.strip():
        logger.info(proc.stdout)

    if proc.returncode != 0:
        raise RuntimeError(f"[matcher] {label} failed")


# --------------------------------------------------
# Matching Statistics
# --------------------------------------------------
def collect_match_stats(database: Path, logger):
    conn = sqlite3.connect(database)
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM images")
    image_count = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM matches")
    pair_count = cur.fetchone()[0]

    cur.execute("SELECT LENGTH(data) FROM matches WHERE LENGTH(data) > 150")
    strong_matches = len(cur.fetchall())

    conn.close()

    expected_pairs = comb(image_count, 2) if image_count >= 2 else 0
    coverage = (strong_matches / pair_count * 100) if pair_count else 0.0

    logger.info(
        f"[matcher] Images: {image_count}, "
        f"Pairs: {pair_count}, "
        f"Strong matches: {strong_matches}, "
        f"Coverage: {coverage:.2f}%"
    )

    return {
        "images": image_count,
        "match_pairs": pair_count,
        "expected_pairs": expected_pairs,
        "good_matches": strong_matches,
        "coverage_percent": coverage,
    }


# --------------------------------------------------
# Pipeline Stage
# --------------------------------------------------
def run(run_root: Path, project_root: Path, force: bool, logger):
    paths = ProjectPaths(project_root)
    paths.ensure_all()

    logger.info("[matcher] Starting exhaustive matching")

    # Load runtime configuration
    config = create_runtime_config(run_root, project_root, logger)
    validate_config(config, logger)

    db = paths.database / "database.db"
    if not db.exists():
        raise FileNotFoundError(f"[matcher] Missing database: {db}")

    if force:
        logger.info("[matcher] Force enabled — clearing previous matches")
        conn = sqlite3.connect(db)
        cur = conn.cursor()
        cur.execute("DELETE FROM matches")
        cur.execute("DELETE FROM two_view_geometries")
        conn.commit()
        conn.close()

    # COLMAP exhaustive matcher
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
        "Exhaustive Matching",
    )

    # Assessment
    stats = collect_match_stats(db, logger)
    assessment = "good" if stats["coverage_percent"] >= 55 else "poor"

    report = {
        "strategy": "exhaustive",
        "statistics": stats,
        "assessment": assessment,
    }

    report_path = paths.database / "matching_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    logger.info(f"[matcher] Report written: {report_path}")
    logger.info("[matcher] Matching stage complete")
