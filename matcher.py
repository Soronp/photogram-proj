#!/usr/bin/env python3
"""
matcher.py

MARK-2 Matching Stage (Project-Scoped DB, GPU-SAFE)
--------------------------------------------------
- Uses project_root/database/database.db
- ToolRunner enforced
- Explicitly disables GPU (COLMAP limitation)
- Resume-safe
"""

import sqlite3
import json
from math import comb
from pathlib import Path

from utils.paths import ProjectPaths
from config_manager import load_config
from tool_runner import ToolRunner


def run(run_root: Path, project_root: Path, force: bool, logger):
    logger.info("[matcher] START")

    # --------------------------------------------------
    # Project-scoped database (INTENTIONAL)
    # --------------------------------------------------
    paths = ProjectPaths(project_root)
    paths.ensure_all()

    db_path = paths.database / "database.db"
    if not db_path.exists():
        raise FileNotFoundError(
            f"[matcher] database.db missing at {db_path}"
        )

    # --------------------------------------------------
    # Load run-scoped config
    # --------------------------------------------------
    config = load_config(run_root, logger)

    # 🔥 CRITICAL FIX:
    # COLMAP exhaustive_matcher does NOT support GPU flags
    original_gpu = config["execution"].get("use_gpu", True)
    config["execution"]["use_gpu"] = False

    try:
        tool = ToolRunner(config, logger)

        # --------------------------------------------------
        # Optional reset
        # --------------------------------------------------
        if force:
            logger.info("[matcher] Force enabled — clearing matches")
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            cur.execute("DELETE FROM matches")
            cur.execute("DELETE FROM two_view_geometries")
            conn.commit()
            conn.close()

        m = config["matching"]

        # --------------------------------------------------
        # Run COLMAP matcher (CPU-only)
        # --------------------------------------------------
        tool.run(
            tool="colmap",
            args=[
                "exhaustive_matcher",
                "--database_path", str(db_path),
                "--SiftMatching.max_ratio", str(m.get("max_ratio", 0.8)),
                "--SiftMatching.max_distance", str(m.get("max_distance", 0.7)),
            ],
        )

    finally:
        # Restore GPU policy for later stages
        config["execution"]["use_gpu"] = original_gpu

    # --------------------------------------------------
    # Coverage report
    # --------------------------------------------------
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM images")
    images = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM matches")
    pairs = cur.fetchone()[0]

    conn.close()

    coverage = (pairs / comb(images, 2) * 100) if images > 1 else 0.0

    report = {
        "method": "exhaustive",
        "scope": "project",
        "images": images,
        "pairs": pairs,
        "coverage_percent": round(coverage, 2),
    }

    report_path = paths.database / "matching_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    logger.info(
        f"[matcher] Images={images}, Pairs={pairs}, Coverage={coverage:.2f}%"
    )
    logger.info("[matcher] COMPLETED")
