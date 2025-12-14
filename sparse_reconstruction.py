#!/usr/bin/env python3
"""
sparse_reconstruction.py

MARK-2 Sparse Reconstruction Stage
----------------------------------
- Uses preprocessed images (images_processed)
- Uses existing COLMAP database (features + matches already computed)
- Runs global SfM via GLOMAP (preferred)
- Produces sparse/0/ model
- Deterministic, logged, restart-safe
"""

import argparse
import shutil
import subprocess
from pathlib import Path

from utils.logger import get_logger
from utils.paths import ProjectPaths
from utils.config import load_config, COLMAP_EXE, GLOMAP_EXE


# ------------------------------------------------------------------
# Command runner
# ------------------------------------------------------------------
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
        logger.error(f"Command failed ({label}): {e}")
        logger.error(e.stdout)
        raise RuntimeError(f"{label} failed")


# ------------------------------------------------------------------
# Sparse model finder
# ------------------------------------------------------------------
def find_sparse_model(root: Path) -> Path | None:
    """Return first subdirectory containing a valid sparse model"""
    for d in root.iterdir():
        if not d.is_dir():
            continue
        required = ["cameras.bin", "images.bin", "points3D.bin"]
        if all((d / r).exists() for r in required):
            return d
    return None


# ------------------------------------------------------------------
# Sparse reconstruction
# ------------------------------------------------------------------
def run_sparse_reconstruction(project_root: Path, force: bool):
    paths = ProjectPaths(project_root)
    config = load_config(project_root)
    logger = get_logger("sparse_reconstruction", project_root)

    # Use preprocessed images
    images_dir = paths.images_processed
    db_path = paths.database / "database.db"
    sparse_root = paths.sparse

    logger.info("Starting sparse reconstruction stage")
    logger.info(f"Preprocessed images: {images_dir}")
    logger.info(f"Database path: {db_path}")
    logger.info(f"Sparse output root: {sparse_root}")

    if not images_dir.exists() or not any(images_dir.iterdir()):
        raise FileNotFoundError(
            f"Preprocessed images folder is empty or missing: {images_dir}"
        )

    if not db_path.exists():
        raise FileNotFoundError(f"COLMAP database not found at: {db_path}")

    if sparse_root.exists():
        if force:
            logger.warning("Removing existing sparse directory (--force)")
            shutil.rmtree(sparse_root)
        else:
            logger.info("Sparse directory already exists — skipping")
            return

    sparse_root.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # Global Sparse Reconstruction (GLOMAP)
    # --------------------------------------------------
    run_command(
        [
            GLOMAP_EXE,
            "mapper",
            "--database_path", db_path,
            "--image_path", images_dir,
            "--output_path", sparse_root
        ],
        logger,
        "Sparse Reconstruction (GLOMAP)"
    )

    # Check that a valid model was created
    model = find_sparse_model(sparse_root)
    if model is None:
        raise RuntimeError("Sparse reconstruction failed — no valid model found")

    logger.info(f"Sparse model generated at: {model}")

    # --------------------------------------------------
    # Convert to PLY for inspection / evaluation
    # --------------------------------------------------
    ply_out = sparse_root / "sparse.ply"
    run_command(
        [
            COLMAP_EXE,
            "model_converter",
            "--input_path", model,
            "--output_path", ply_out,
            "--output_type", "PLY"
        ],
        logger,
        "Sparse Model Conversion"
    )

    logger.info("Sparse reconstruction completed successfully")
    logger.info(f"PLY output: {ply_out}")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="MARK-2 Sparse Reconstruction")
    parser.add_argument("project_root", type=Path)
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing sparse output"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_sparse_reconstruction(args.project_root, args.force)


if __name__ == "__main__":
    main()
