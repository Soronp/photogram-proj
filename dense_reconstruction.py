#!/usr/bin/env python3
"""
dense_reconstruction.py

MARK-2 Dense Reconstruction (Hybrid-Aware)
------------------------------------------
- Consumes GLOMAP or COLMAP sparse models
- Dynamically adapts COLMAP MVS parameters
- Streams stdout (no silent hangs)
- Deterministic, restart-safe
"""

import argparse
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Optional

from utils.logger import get_logger
from utils.paths import ProjectPaths
from utils.config import load_config, COLMAP_EXE


# ------------------------------------------------------------------
# Streaming command runner (parity-safe)
# ------------------------------------------------------------------
def run_command(cmd, logger, label):
    cmd = [str(c) for c in cmd]
    logger.info(f"[RUN] {label}")
    logger.info(" ".join(cmd))

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    for line in process.stdout:
        logger.info(line.rstrip())

    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"{label} failed")


# ------------------------------------------------------------------
# Sparse model discovery (GLOMAP/COLMAP agnostic)
# ------------------------------------------------------------------
def find_sparse_model(sparse_root: Path) -> Optional[Path]:
    required = {"cameras.bin", "images.bin", "points3D.bin"}

    for d in sorted(sparse_root.iterdir()):
        if d.is_dir() and required.issubset({f.name for f in d.iterdir()}):
            return d
    return None


# ------------------------------------------------------------------
# Sparse analysis (hybrid-aware, CLI-safe)
# ------------------------------------------------------------------
def analyze_sparse_model(model_dir: Path, images_dir: Path, logger) -> Dict:
    image_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    total_images = sum(1 for f in images_dir.iterdir() if f.suffix.lower() in image_exts)

    images_bin = model_dir / "images.bin"
    points_bin = model_dir / "points3D.bin"

    reconstructed = images_bin.stat().st_size // 64 if images_bin.exists() else 0
    points = points_bin.stat().st_size // 24 if points_bin.exists() else 0

    coverage = (reconstructed / total_images * 100) if total_images else 0

    if coverage >= 70 and points > 15000:
        quality = "good"
    elif coverage >= 40:
        quality = "fair"
    else:
        quality = "poor"

    logger.info("Sparse Analysis:")
    logger.info(f"  Images used  : {reconstructed}/{total_images}")
    logger.info(f"  Points      : {points}")
    logger.info(f"  Coverage    : {coverage:.1f}%")
    logger.info(f"  Quality     : {quality}")

    return {
        "quality": quality,
        "coverage": coverage,
        "points": points,
    }


# ------------------------------------------------------------------
# Dynamic dense parameter selection
# ------------------------------------------------------------------
def select_dense_parameters(analysis: Dict, logger) -> Dict:
    q = analysis["quality"]

    if q == "poor":
        params = {
            "max_image_size": "1400",
            "patchmatch": [
                "--PatchMatchStereo.geom_consistency", "false",
                "--PatchMatchStereo.filter", "true",
                "--PatchMatchStereo.num_iterations", "2",
                "--PatchMatchStereo.num_samples", "10",
                "--PatchMatchStereo.cache_size", "32",
            ],
        }
    elif q == "fair":
        params = {
            "max_image_size": "2000",
            "patchmatch": [
                "--PatchMatchStereo.geom_consistency", "true",
                "--PatchMatchStereo.filter", "true",
                "--PatchMatchStereo.num_iterations", "3",
                "--PatchMatchStereo.num_samples", "15",
                "--PatchMatchStereo.cache_size", "32",
            ],
        }
    else:
        params = {
            "max_image_size": "2400",
            "patchmatch": [
                "--PatchMatchStereo.geom_consistency", "true",
                "--PatchMatchStereo.filter", "true",
                "--PatchMatchStereo.num_iterations", "5",
                "--PatchMatchStereo.num_samples", "25",
                "--PatchMatchStereo.cache_size", "32",
            ],
        }

    logger.info(f"Dense preset selected: {q}")
    return params


# ------------------------------------------------------------------
# Execution
# ------------------------------------------------------------------
def run_dense_reconstruction(project_root: Path, force: bool):
    paths = ProjectPaths(project_root)
    load_config(project_root)
    logger = get_logger("dense_reconstruction", project_root)

    images_dir = paths.images_processed
    sparse_root = paths.sparse
    dense_root = paths.dense
    fused_ply = dense_root / "fused.ply"

    model_dir = find_sparse_model(sparse_root)
    if model_dir is None:
        raise RuntimeError("No valid sparse model found")

    if fused_ply.exists() and not force:
        logger.info("Dense output exists — skipping")
        return

    if dense_root.exists():
        shutil.rmtree(dense_root)
    dense_root.mkdir(parents=True)

    analysis = analyze_sparse_model(model_dir, images_dir, logger)
    params = select_dense_parameters(analysis, logger)

    # Undistortion
    run_command([
        COLMAP_EXE, "image_undistorter",
        "--image_path", images_dir,
        "--input_path", model_dir,
        "--output_path", dense_root,
        "--output_type", "COLMAP",
        "--max_image_size", params["max_image_size"],
    ], logger, "Image Undistortion")

    # PatchMatch
    cmd = [
        COLMAP_EXE, "patch_match_stereo",
        "--workspace_path", dense_root,
        "--workspace_format", "COLMAP",
    ] + params["patchmatch"]

    run_command(cmd, logger, "PatchMatch Stereo")

    # Fusion
    run_command([
        COLMAP_EXE, "stereo_fusion",
        "--workspace_path", dense_root,
        "--workspace_format", "COLMAP",
        "--input_type", "geometric",
        "--output_path", fused_ply,
    ], logger, "Stereo Fusion")

    if not fused_ply.exists() or fused_ply.stat().st_size < 100_000:
        raise RuntimeError("Dense reconstruction failed")

    logger.info("Dense reconstruction completed successfully")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("project_root", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    run_dense_reconstruction(args.project_root, args.force)


if __name__ == "__main__":
    main()
