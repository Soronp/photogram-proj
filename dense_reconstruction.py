#!/usr/bin/env python3
"""
dense_reconstruction.py

MARK-2 Dense Reconstruction Stage
---------------------------------
- Uses sparse reconstruction + processed images
- Runs COLMAP dense pipeline (undistort → PatchMatch → fusion)
- Produces dense/fused.ply
- Fully logged, deterministic, restart-safe
"""

import argparse
import shutil
import subprocess
from pathlib import Path

from utils.logger import get_logger
from utils.paths import ProjectPaths
from utils.config import load_config, COLMAP_EXE


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
# Dense reconstruction
# ------------------------------------------------------------------
def run_dense_reconstruction(project_root: Path, force: bool):
    paths = ProjectPaths(project_root)
    config = load_config(project_root)
    logger = get_logger("dense_reconstruction", project_root)

    images_dir = paths.images_processed  # Correct folder
    sparse_root = paths.sparse
    dense_root = paths.dense

    logger.info("Starting dense reconstruction stage")
    logger.info(f"Images: {images_dir}")
    logger.info(f"Sparse model root: {sparse_root}")
    logger.info(f"Dense output root: {dense_root}")

    if not images_dir.exists() or not any(images_dir.iterdir()):
        raise FileNotFoundError(f"No images found in {images_dir}")

    # Locate sparse model (expects sparse/0/)
    model_dir = next(
        (d for d in sparse_root.iterdir() if d.is_dir() and (d / "cameras.bin").exists()),
        None
    )
    if model_dir is None:
        raise RuntimeError("No valid sparse model found for dense reconstruction")

    logger.info(f"Using sparse model: {model_dir}")

    # Prepare dense directory
    if dense_root.exists():
        if force:
            logger.warning("Removing existing dense directory (--force)")
            shutil.rmtree(dense_root)
        else:
            logger.info("Dense directory already exists — skipping stage")
            return

    dense_root.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # 1. Image Undistortion (COLMAP format)
    # --------------------------------------------------
    max_size = int(config.get("dense_max_image_size", 2800))

    run_command(
        [
            COLMAP_EXE,
            "image_undistorter",
            "--image_path", str(images_dir),
            "--input_path", str(model_dir),
            "--output_path", str(dense_root),
            "--output_type", "COLMAP",
            "--max_image_size", str(max_size)
        ],
        logger,
        "Image Undistortion"
    )

    undistorted_images = dense_root / "images"
    if not undistorted_images.exists() or not any(undistorted_images.iterdir()):
        raise RuntimeError("No undistorted images produced")

    logger.info(f"Undistorted images ready: {len(list(undistorted_images.iterdir()))} images")

    # --------------------------------------------------
    # 2. PatchMatch Stereo (geometric consistency)
    # --------------------------------------------------
    run_command(
        [
            COLMAP_EXE,
            "patch_match_stereo",
            "--workspace_path", str(dense_root),
            "--workspace_format", "COLMAP",
            "--PatchMatchStereo.geom_consistency", "1",
            "--PatchMatchStereo.filter", "1",
            "--PatchMatchStereo.window_radius", "5",
            "--PatchMatchStereo.num_samples", "15",
            "--PatchMatchStereo.num_iterations", "3"
        ],
        logger,
        "PatchMatch Stereo"
    )

    depth_maps = dense_root / "stereo" / "depth_maps"
    if not depth_maps.exists() or not any(depth_maps.iterdir()):
        raise RuntimeError("PatchMatch produced no depth maps")

    # --------------------------------------------------
    # 3. Stereo Fusion
    # --------------------------------------------------
    fused_ply = dense_root / "fused.ply"

    run_command(
        [
            COLMAP_EXE,
            "stereo_fusion",
            "--workspace_path", str(dense_root),
            "--workspace_format", "COLMAP",
            "--input_type", "geometric",
            "--output_path", str(fused_ply)
        ],
        logger,
        "Stereo Fusion"
    )

    if not fused_ply.exists() or fused_ply.stat().st_size == 0:
        raise RuntimeError("Fusion failed: fused.ply is empty")

    logger.info("Dense reconstruction completed successfully")
    logger.info(f"Dense point cloud: {fused_ply}")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="MARK-2 Dense Reconstruction")
    parser.add_argument("project_root", type=Path)
    parser.add_argument("--force", action="store_true", help="Overwrite existing dense output")
    args = parser.parse_args()

    run_dense_reconstruction(args.project_root, args.force)


if __name__ == "__main__":
    main()
