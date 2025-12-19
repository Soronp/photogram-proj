#!/usr/bin/env python3
"""
openmvs_export.py

MARK-2 OpenMVS Export (Pipeline-Owned Sparse Input)
--------------------------------------------------
- Uses MARK-2 canonical ProjectPaths
- Consumes sparse reconstruction from project_root/sparse/*
- Produces openmvs/scene.mvs deterministically
"""

from pathlib import Path
import shutil
import subprocess

from utils.logger import get_logger
from utils.paths import ProjectPaths
from utils.config import COLMAP_EXE


def run(cmd, log, label, cwd: Path):
    cmd = [str(c) for c in cmd]
    log.info(f"[RUN:{label}] {' '.join(cmd)}")

    proc = subprocess.run(
        cmd,
        cwd=cwd,                       # 🔴 CRITICAL FIX
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    log.info(proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(f"{label} failed")


def find_sparse_model(sparse_root: Path) -> Path:
    """
    Finds a valid COLMAP sparse model inside sparse_root.
    Assumes standard COLMAP layout: sparse/0/, sparse/1/, etc.
    """
    if not sparse_root.exists():
        raise RuntimeError(f"Sparse root missing: {sparse_root}")

    candidates = []
    for d in sorted(sparse_root.iterdir()):
        if not d.is_dir():
            continue
        files = {f.name for f in d.iterdir()}
        if {"cameras.bin", "images.bin", "points3D.bin"}.issubset(files):
            candidates.append(d)

    if not candidates:
        raise RuntimeError("No valid sparse model found in sparse/")

    if len(candidates) > 1:
        raise RuntimeError(
            f"Multiple sparse models found: {candidates}. "
            "MARK-2 requires one active sparse model."
        )

    return candidates[0]


def run_openmvs_export(project_root: Path, force: bool):
    project_root = Path(project_root).resolve()
    paths = ProjectPaths(project_root)
    paths.ensure_all()
    paths.validate()

    log = get_logger("openmvs_export", project_root)
    log.info("Starting OpenMVS export")

    # --------------------------------------------------
    # Locate sparse model (pipeline-owned)
    # --------------------------------------------------
    sparse_model = find_sparse_model(paths.sparse)
    log.info(f"Using sparse model: {sparse_model}")

    # --------------------------------------------------
    # Prepare undistorted workspace
    # --------------------------------------------------
    undistorted = paths.openmvs / "undistorted"
    if undistorted.exists() and force:
        shutil.rmtree(undistorted)

    undistorted.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # COLMAP image undistortion
    # --------------------------------------------------
    run(
        [
            COLMAP_EXE, "image_undistorter",
            "--image_path", paths.images_processed,
            "--input_path", sparse_model,
            "--output_path", undistorted,
            "--output_type", "COLMAP",
        ],
        log,
        "COLMAP image_undistorter",
        cwd=project_root,
    )

    # Remove stereo folder (OpenMVS requirement)
    stereo = undistorted / "stereo"
    if stereo.exists():
        shutil.rmtree(stereo)

    # Validate undistorted sparse
    undistorted_sparse = undistorted / "sparse"
    required = {"cameras.bin", "images.bin", "points3D.bin"}

    if not undistorted_sparse.exists():
        raise RuntimeError("Undistorted sparse folder missing")

    if not required.issubset({f.name for f in undistorted_sparse.iterdir()}):
        raise RuntimeError("Undistorted sparse is invalid")

    # --------------------------------------------------
    # InterfaceCOLMAP → scene.mvs
    # --------------------------------------------------
    scene = paths.openmvs_scene
    if scene.exists() and force:
        scene.unlink()

    run(
        [
            "InterfaceCOLMAP",
            "-i", undistorted,
            "-o", scene,
            "--image-folder", undistorted / "images",
        ],
        log,
        "InterfaceCOLMAP",
        cwd=project_root,
    )

    if not scene.exists() or scene.stat().st_size < 50_000:
        raise RuntimeError("scene.mvs not generated or invalid")

    log.info(f"OpenMVS scene successfully created: {scene}")
