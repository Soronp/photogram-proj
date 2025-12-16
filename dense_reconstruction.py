#!/usr/bin/env python3
"""
dense_reconstruction.py

MARK-2 Dense Reconstruction (OpenMVS-first)
-------------------------------------------
- Primary backend: OpenMVS
- Fallback backend: COLMAP MVS
- Deterministic, restart-safe
- Guarantees dense/fused.ply output
"""

import argparse
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from utils.logger import get_logger
from utils.paths import ProjectPaths
from utils.config import load_config, COLMAP_EXE


# --------------------------------------------------
# Utility
# --------------------------------------------------

def command_exists(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def run_command(cmd, logger, label):
    cmd = [str(c) for c in cmd]
    logger.info(f"[RUN] {label}")
    logger.info(" ".join(cmd))

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    for line in proc.stdout:
        logger.info(line.rstrip())

    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"{label} failed")


# --------------------------------------------------
# Sparse model discovery
# --------------------------------------------------

def find_sparse_model(sparse_root: Path) -> Optional[Path]:
    required = {"cameras.bin", "images.bin", "points3D.bin"}
    for d in sorted(sparse_root.iterdir()):
        if d.is_dir() and required.issubset({f.name for f in d.iterdir()}):
            return d
    return None


# --------------------------------------------------
# OpenMVS backend
# --------------------------------------------------

def run_openmvs(
    model_dir: Path,
    images_dir: Path,
    dense_root: Path,
    fused_ply: Path,
    logger,
):
    logger.info("Using OpenMVS backend")

    openmvs_dir = dense_root / "openmvs"
    openmvs_dir.mkdir(parents=True, exist_ok=True)

    scene_mvs = openmvs_dir / "scene.mvs"
    dense_mvs = openmvs_dir / "dense.mvs"

    # Step 1: InterfaceCOLMAP
    run_command([
        "InterfaceCOLMAP",
        "--input-path", model_dir,
        "--image-path", images_dir,
        "--output-file", scene_mvs,
    ], logger, "OpenMVS InterfaceCOLMAP")

    # Step 2: DensifyPointCloud
    run_command([
        "DensifyPointCloud",
        scene_mvs,
        "--output-file", dense_mvs,
        "--resolution-level", "2",
        "--number-views", "8",
    ], logger, "OpenMVS DensifyPointCloud")

    # Step 3: Export dense point cloud
    run_command([
        "ExportPointCloud",
        dense_mvs,
        "--output-file", fused_ply,
    ], logger, "OpenMVS ExportPointCloud")

    if not fused_ply.exists() or fused_ply.stat().st_size < 100_000:
        raise RuntimeError("OpenMVS output invalid")


# --------------------------------------------------
# COLMAP fallback backend
# --------------------------------------------------

def run_colmap_mvs(
    model_dir: Path,
    images_dir: Path,
    dense_root: Path,
    fused_ply: Path,
    logger,
):
    logger.warning("Falling back to COLMAP dense reconstruction")

    # Undistort
    run_command([
        COLMAP_EXE, "image_undistorter",
        "--image_path", images_dir,
        "--input_path", model_dir,
        "--output_path", dense_root,
        "--output_type", "COLMAP",
        "--max_image_size", "2000",
    ], logger, "COLMAP Image Undistortion")

    # PatchMatch
    run_command([
        COLMAP_EXE, "patch_match_stereo",
        "--workspace_path", dense_root,
        "--workspace_format", "COLMAP",
        "--PatchMatchStereo.geom_consistency", "true",
    ], logger, "COLMAP PatchMatchStereo")

    # Fusion
    run_command([
        COLMAP_EXE, "stereo_fusion",
        "--workspace_path", dense_root,
        "--workspace_format", "COLMAP",
        "--input_type", "geometric",
        "--output_path", fused_ply,
    ], logger, "COLMAP StereoFusion")

    if not fused_ply.exists() or fused_ply.stat().st_size < 100_000:
        raise RuntimeError("COLMAP dense output invalid")


# --------------------------------------------------
# Main dense stage
# --------------------------------------------------

def run_dense_reconstruction(project_root: Path, force: bool):
    load_config(project_root)
    paths = ProjectPaths(project_root)
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
    dense_root.mkdir(parents=True, exist_ok=True)

    openmvs_available = (
        command_exists("InterfaceCOLMAP")
        and command_exists("DensifyPointCloud")
        and command_exists("ExportPointCloud")
    )

    if openmvs_available:
        try:
            run_openmvs(model_dir, images_dir, dense_root, fused_ply, logger)
            logger.info("Dense reconstruction completed with OpenMVS")
            return
        except Exception as e:
            logger.error(f"OpenMVS failed: {e}")

    # Fallback
    run_colmap_mvs(model_dir, images_dir, dense_root, fused_ply, logger)
    logger.info("Dense reconstruction completed with COLMAP fallback")


# --------------------------------------------------
# CLI
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MARK-2 Dense Reconstruction")
    parser.add_argument("project_root", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    run_dense_reconstruction(args.project_root, args.force)


if __name__ == "__main__":
    main()
