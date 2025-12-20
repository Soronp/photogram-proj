#!/usr/bin/env python3
"""
dense_reconstruction.py

MARK-2 Dense Reconstruction Stage (OpenMVS)
-------------------------------------------
Runner-managed, deterministic, non-interactive.
"""

import subprocess
from pathlib import Path

from utils.paths import ProjectPaths


# --------------------------------------------------
# PIPELINE STAGE
# --------------------------------------------------
def run(project_root: Path, force: bool, logger):
    paths = ProjectPaths(project_root)
    paths.ensure_all()

    logger.info("[dense] Stage started (OpenMVS)")

    openmvs_root = paths.openmvs
    scene_file = openmvs_root / "scene.mvs"
    undistorted_dir = openmvs_root / "undistorted"

    if not scene_file.exists():
        raise FileNotFoundError(f"scene.mvs not found: {scene_file}")

    if not undistorted_dir.exists():
        raise FileNotFoundError(f"undistorted/ not found: {undistorted_dir}")

    paths.dense.mkdir(parents=True, exist_ok=True)
    fused_file = paths.dense / "fused.ply"

    if fused_file.exists() and not force:
        logger.info("[dense] Output exists — skipping")
        return

    cmd = [
        "DensifyPointCloud",
        "-i", str(scene_file),
        "-o", str(fused_file),
        "--working-folder", str(openmvs_root),
        "--resolution-level", "1",
        "--max-resolution", "2560",
        "--min-resolution", "640",
        "--number-views", "8",
        "--number-views-fuse", "3",
        "--estimate-colors", "2",
        "--estimate-normals", "2",
        "--filter-point-cloud", "1",
    ]

    logger.info("[dense] RUN: DensifyPointCloud")
    logger.info(" ".join(cmd))

    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError("DensifyPointCloud failed")

    logger.info(f"[dense] Output written: {fused_file}")
    logger.info("[dense] Stage completed")
