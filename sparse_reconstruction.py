#!/usr/bin/env python3
"""
sparse_reconstruction.py

MARK-2 Sparse Reconstruction Stage (CONVENTION-COMPATIBLE)
---------------------------------------------------------
- Uses preprocessed images (images_processed)
- Uses existing COLMAP database (features + matches already computed)
- Chooses GLOMAP or COLMAP deterministically
- Produces sparse/<model_id>/ model
- Converts model to PLY for inspection
- Deterministic, logged, restart-safe, runner-compatible
- STRICTLY follows MARK-2 conventions used by dense / mesh stages
"""

import argparse
import shutil
import subprocess
import sqlite3
from pathlib import Path
from typing import Dict, Optional

from utils.logger import get_logger
from utils.paths import ProjectPaths
from utils.config import load_config, COLMAP_EXE, GLOMAP_EXE


# ------------------------------------------------------------------
# Command runner (IDENTICAL CONTRACT TO DENSE STAGE)
# ------------------------------------------------------------------

def run_command(cmd, logger, label: str):
    cmd = [str(c) for c in cmd]
    logger.info(f"[RUN] {label}")
    logger.info(" ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        logger.info(result.stdout)
    except subprocess.CalledProcessError as e:
        logger.error(f"[FAIL] {label}")
        logger.error(e.stdout)
        raise RuntimeError(f"{label} failed") from e


# ------------------------------------------------------------------
# Sparse model discovery (FILESYSTEM CONTRACT)
# ------------------------------------------------------------------

def find_sparse_model(sparse_root: Path) -> Optional[Path]:
    if not sparse_root.exists():
        return None

    required = {"cameras.bin", "images.bin", "points3D.bin"}

    for d in sorted(sparse_root.iterdir()):
        if not d.is_dir():
            continue
        if required.issubset({f.name for f in d.iterdir()}):
            return d

    return None


# ------------------------------------------------------------------
# Dataset analysis (SAFE SQLITE INSPECTION ONLY)
# ------------------------------------------------------------------

def analyze_dataset(db_path: Path, logger) -> Dict:
    logger.info("Analyzing dataset characteristics")

    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()

        cur.execute("SELECT COUNT(*) FROM images")
        image_count = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM matches")
        match_pair_count = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM matches WHERE LENGTH(data) > 100")
        good_match_count = cur.fetchone()[0]

    expected_pairs = (image_count * (image_count - 1)) // 2

    match_coverage = (
        (match_pair_count / expected_pairs) * 100
        if expected_pairs > 0
        else 0.0
    )

    good_match_ratio = (
        (good_match_count / match_pair_count) * 100
        if match_pair_count > 0
        else 0.0
    )

    if good_match_ratio >= 80:
        match_quality = "excellent"
    elif good_match_ratio >= 60:
        match_quality = "good"
    elif good_match_ratio >= 40:
        match_quality = "fair"
    else:
        match_quality = "poor"

    logger.info(f"Images            : {image_count}")
    logger.info(f"Match pairs       : {match_pair_count}/{expected_pairs} ({match_coverage:.1f}%)")
    logger.info(f"Good match ratio  : {good_match_ratio:.1f}%")
    logger.info(f"Match quality     : {match_quality}")

    return {
        "image_count": image_count,
        "match_quality": match_quality,
    }


# ------------------------------------------------------------------
# Parameter selection (STATIC, VERSION-SAFE)
# ------------------------------------------------------------------

def select_strategy(analysis: Dict, logger) -> Dict:
    image_count = analysis["image_count"]
    match_quality = analysis["match_quality"]

    use_glomap = image_count <= 30 and match_quality in {"good", "excellent"}

    if use_glomap:
        logger.info("Strategy: GLOMAP (small, well-matched dataset)")
        return {"use_glomap": True}

    logger.info("Strategy: COLMAP Mapper")

    params = []

    if match_quality == "poor":
        params += [
            "--Mapper.init_min_tri_angle", "1",
            "--Mapper.min_num_matches", "8",
        ]
    elif match_quality == "fair":
        params += [
            "--Mapper.init_min_tri_angle", "2",
            "--Mapper.min_num_matches", "10",
        ]
    else:
        params += [
            "--Mapper.init_min_tri_angle", "4",
            "--Mapper.min_num_matches", "15",
        ]

    params += [
        "--Mapper.multiple_models", "0",
        "--Mapper.max_num_models", "1",
    ]

    return {
        "use_glomap": False,
        "colmap_params": params,
    }


# ------------------------------------------------------------------
# Sparse reconstruction entrypoint
# ------------------------------------------------------------------

def run_sparse_reconstruction(project_root: Path, force: bool):
    load_config(project_root)
    paths = ProjectPaths(project_root)
    logger = get_logger("sparse_reconstruction", project_root)

    images_dir = paths.images_processed
    db_path = paths.database / "database.db"
    sparse_root = paths.sparse

    logger.info("Starting sparse reconstruction")
    logger.info(f"Images directory : {images_dir}")
    logger.info(f"Database path    : {db_path}")
    logger.info(f"Sparse root      : {sparse_root}")
    logger.info(f"Force rebuild    : {force}")

    if not images_dir.exists() or not any(images_dir.iterdir()):
        raise FileNotFoundError(f"Images missing: {images_dir}")

    if not db_path.exists():
        raise FileNotFoundError(f"Database missing: {db_path}")

    existing = find_sparse_model(sparse_root)
    if existing and not force:
        logger.info("Sparse model exists - skipping")
        return

    if sparse_root.exists():
        shutil.rmtree(sparse_root)
    sparse_root.mkdir(parents=True, exist_ok=True)

    analysis = analyze_dataset(db_path, logger)
    strategy = select_strategy(analysis, logger)

    if strategy["use_glomap"]:
        _run_glomap(db_path, images_dir, sparse_root, logger)
    else:
        _run_colmap(db_path, images_dir, sparse_root, strategy["colmap_params"], logger)

    model_dir = find_sparse_model(sparse_root)
    if model_dir is None:
        raise RuntimeError("Sparse reconstruction failed")

    logger.info(f"Sparse model created: {model_dir}")

    _convert_to_ply(model_dir, sparse_root, logger)

    logger.info("Sparse reconstruction completed")


# ------------------------------------------------------------------
# Execution helpers (MATCH DENSE STAGE STYLE)
# ------------------------------------------------------------------

def _run_glomap(db_path: Path, images_dir: Path, sparse_root: Path, logger):
    try:
        run_command(
            [
                GLOMAP_EXE,
                "mapper",
                "--database_path", db_path,
                "--image_path", images_dir,
                "--output_path", sparse_root,
            ],
            logger,
            "GLOMAP Mapper",
        )
    except Exception as e:
        logger.warning(f"GLOMAP failed: {e}")
        logger.info("Falling back to COLMAP")

        if sparse_root.exists():
            shutil.rmtree(sparse_root)
        sparse_root.mkdir(parents=True, exist_ok=True)

        _run_colmap(db_path, images_dir, sparse_root, [], logger)


def _run_colmap(db_path: Path, images_dir: Path, sparse_root: Path, params, logger):
    cmd = [
        COLMAP_EXE,
        "mapper",
        "--database_path", db_path,
        "--image_path", images_dir,
        "--output_path", sparse_root,
    ] + params

    run_command(cmd, logger, "COLMAP Mapper")


def _convert_to_ply(model_dir: Path, sparse_root: Path, logger):
    ply_path = sparse_root / "sparse.ply"

    run_command(
        [
            COLMAP_EXE,
            "model_converter",
            "--input_path", model_dir,
            "--output_path", ply_path,
            "--output_type", "PLY",
        ],
        logger,
        "Sparse PLY Conversion",
    )


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MARK-2 Sparse Reconstruction")
    parser.add_argument("project_root", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    run_sparse_reconstruction(args.project_root, args.force)


if __name__ == "__main__":
    main()
