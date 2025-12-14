#!/usr/bin/env python3
"""
matcher.py

MARK-2 Matching Stage
--------------------
- Uses exhaustive matching ONLY (deterministic, robust)
- Populates match tables in COLMAP database
- Prevents partial match graphs caused by sequential matching

Reads:
- database/database.db

Writes:
- database/database.db (matches)
- database/matching_report.json
- logs/matcher.log
"""

import argparse
import subprocess
import json
from pathlib import Path

from utils.logger import get_logger
from utils.paths import ProjectPaths
from utils.config import load_config


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="MARK-2 COLMAP exhaustive matcher")
    parser.add_argument(
        "--project",
        required=True,
        help="Path to project root"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run matching (clears existing matches if present)"
    )
    return parser.parse_args()


# ------------------------------------------------------------------
# Command runner
# ------------------------------------------------------------------

def run(cmd, logger):
    logger.info("[RUN] " + " ".join(cmd))
    subprocess.run(cmd, check=True)


# ------------------------------------------------------------------
# Main logic
# ------------------------------------------------------------------

def main():
    args = parse_args()
    project_root = Path(args.project).resolve()

    paths = ProjectPaths(project_root)
    paths.ensure_all()

    logger = get_logger("matcher", project_root)
    _ = load_config(project_root)  # loaded for parity / future use

    db_path = paths.database / "database.db"
    if not db_path.exists():
        raise FileNotFoundError(
            "database.db not found — run database_builder.py first"
        )

    logger.info("Starting exhaustive image matching")
    logger.info(f"Database: {db_path}")

    # NOTE:
    # COLMAP exhaustive_matcher automatically overwrites existing matches.
    # --force is kept for CLI parity and future extension.
    cmd = [
        "colmap",
        "exhaustive_matcher",
        "--database_path", str(db_path),
    ]

    run(cmd, logger)

    # ------------------------------------------------------------------
    # Write matching report
    # ------------------------------------------------------------------

    report = {
        "strategy": "exhaustive",
        "database": str(db_path),
        "note": "Sequential matching intentionally disabled in MARK-2",
    }

    report_path = paths.database / "matching_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Exhaustive matching completed successfully")
    logger.info(f"Matching report: {report_path}")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    main()
