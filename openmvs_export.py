#!/usr/bin/env python3
"""
openmvs_export.py

MARK-2 OpenMVS Export (Contract-Driven)
--------------------------------------
- Reads export_ready.json (no guessing)
- Uses exactly one sparse model
- Produces openmvs/scene.mvs deterministically
"""

import json
import shutil
import subprocess
from pathlib import Path

from utils.logger import get_logger
from utils.paths import ProjectPaths
from utils.config import COLMAP_EXE


REQUIRED = {"cameras.bin", "images.bin", "points3D.bin"}


def run(cmd, log, label, cwd: Path):
    cmd = [str(c) for c in cmd]
    log.info(f"[RUN:{label}] {' '.join(cmd)}")

    proc = subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    log.info(proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(f"{label} failed")


def run_openmvs_export(project_root: Path, force: bool):
    project_root = Path(project_root).resolve()
    paths = ProjectPaths(project_root)
    paths.ensure_all()
    paths.validate()

    log = get_logger("openmvs_export", project_root)
    log.info("Starting OpenMVS export")

    # --------------------------------------------------
    # Read authoritative sparse selection
    # --------------------------------------------------
    meta_path = paths.sparse / "export_ready.json"
    if not meta_path.exists():
        raise RuntimeError("export_ready.json missing (sparse stage not finalized)")

    meta = json.loads(meta_path.read_text())
    model_dir = paths.sparse / meta["model_dir"]

    if not model_dir.exists():
        raise RuntimeError(f"Declared sparse model missing: {model_dir}")

    files = {f.name for f in model_dir.iterdir()}
    if not REQUIRED.issubset(files):
        raise RuntimeError("Declared sparse model is invalid")

    log.info(f"Using sparse model: {model_dir.name}")

    # --------------------------------------------------
    # Prepare undistorted workspace
    # --------------------------------------------------
    undistorted = paths.openmvs / "undistorted"
    if undistorted.exists() and force:
        shutil.rmtree(undistorted)

    undistorted.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # COLMAP image undistorter
    # --------------------------------------------------
    run(
        [
            COLMAP_EXE, "image_undistorter",
            "--image_path", paths.images_processed,
            "--input_path", model_dir,
            "--output_path", undistorted,
            "--output_type", "COLMAP",
        ],
        log,
        "COLMAP image_undistorter",
        cwd=project_root,
    )

    # Remove stereo (OpenMVS requirement)
    stereo = undistorted / "stereo"
    if stereo.exists():
        shutil.rmtree(stereo)

    # Validate undistorted sparse
    undistorted_sparse = undistorted / "sparse"
    if not undistorted_sparse.exists():
        raise RuntimeError("Undistorted sparse folder missing")

    if not REQUIRED.issubset({f.name for f in undistorted_sparse.iterdir()}):
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

    log.info(f"OpenMVS scene created successfully: {scene}")
