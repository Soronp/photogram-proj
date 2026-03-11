#!/usr/bin/env python3
"""
matcher.py

Stage 5 — Feature Matching

Scalable COLMAP matching stage.

Responsibilities
----------------
• select matching strategy based on dataset size
• perform feature matching
• run geometric verification
• compute matching coverage statistics
"""

import sqlite3
import json
from math import comb


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def compute_coverage(images, pairs):

    if images <= 1:
        return 0.0

    total_pairs = comb(images, 2)

    return (pairs / total_pairs) * 100


def get_image_count(db_path):

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM images")
    images = cur.fetchone()[0]

    conn.close()

    return images


def select_matcher(n_images):
    """
    Choose matching strategy based on dataset size.
    """

    if n_images < 150:
        return "exhaustive_matcher"

    if n_images < 800:
        return "sequential_matcher"

    return "vocab_tree_matcher"


# --------------------------------------------------
# Matching execution
# --------------------------------------------------

def run_matching(paths, tools, logger, db_path, matcher, match_cfg):

    max_ratio = match_cfg.get("max_ratio", 0.85)
    max_distance = match_cfg.get("max_distance", 0.8)
    cross_check = match_cfg.get("cross_check", True)

    cmd = [
        matcher,
        "--database_path", str(db_path),

        "--SiftMatching.max_ratio", str(max_ratio),
        "--SiftMatching.max_distance", str(max_distance),

        "--SiftMatching.cross_check",
        "1" if cross_check else "0",
    ]

    if matcher == "sequential_matcher":

        cmd += [
            "--SequentialMatching.overlap",
            str(match_cfg.get("sequential_overlap", 10))
        ]

    logger.info(f"[matcher] running {matcher}")

    tools.run(
        "colmap",
        cmd,
    )


def run_geometric_verification(paths, tools, logger, db_path, config):

    geo_cfg = config.get("geometric_verification", {})

    min_inliers = geo_cfg.get("min_num_inliers", 25)
    max_error = geo_cfg.get("max_error", 3)
    confidence = geo_cfg.get("confidence", 0.9999)

    logger.info("[matcher] running geometric verification")

    tools.run(
        "colmap",
        [
            "geometric_verifier",

            "--database_path", str(db_path),

            "--TwoViewGeometry.min_num_inliers",
            str(min_inliers),

            "--TwoViewGeometry.max_error",
            str(max_error),

            "--TwoViewGeometry.confidence",
            str(confidence),
        ],
    )


# --------------------------------------------------
# Coverage report
# --------------------------------------------------

def write_report(paths, db_path, logger):

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM images")
    images = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM two_view_geometries")
    verified_pairs = cur.fetchone()[0]

    conn.close()

    coverage = compute_coverage(images, verified_pairs)

    report = {
        "images": images,
        "verified_pairs": verified_pairs,
        "coverage_percent": round(coverage, 2)
    }

    report_path = paths.database / "matching_report.json"

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"[matcher] verified pairs: {verified_pairs}")
    logger.info(f"[matcher] coverage: {coverage:.2f}%")

    return report


# --------------------------------------------------
# Stage execution
# --------------------------------------------------

def run(paths, logger, tools, config):

    logger.info("[matcher] stage started")

    db_path = paths.database_path

    if not db_path.exists():
        raise RuntimeError("database.db missing — run db_builder first")

    match_cfg = config.get("matching", {})

    # --------------------------------------------------
    # Determine dataset size
    # --------------------------------------------------

    n_images = get_image_count(db_path)

    logger.info(f"[matcher] dataset size: {n_images} images")

    matcher = select_matcher(n_images)

    logger.info(f"[matcher] selected strategy: {matcher}")

    # --------------------------------------------------
    # Feature matching
    # --------------------------------------------------

    run_matching(
        paths,
        tools,
        logger,
        db_path,
        matcher,
        match_cfg
    )

    # --------------------------------------------------
    # Geometric verification
    # --------------------------------------------------

    run_geometric_verification(
        paths,
        tools,
        logger,
        db_path,
        config
    )

    # --------------------------------------------------
    # Coverage diagnostics
    # --------------------------------------------------

    write_report(paths, db_path, logger)

    logger.info("[matcher] stage completed")