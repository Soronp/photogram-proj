#!/usr/bin/env python3
"""
openmvs_export.py

MARK-2 OpenMVS Export (User-Directed Sparse Input)
-------------------------------------------------
- User provides sparse output root folder
- export_ready.json must exist in that folder
- Sparse model is inside a subfolder (e.g. 0/, 1/)
- Produces OpenMVS scene.mvs deterministically
"""

from pathlib import Path
import json
import shutil
import subprocess

from utils.logger import get_logger
from utils.paths import ProjectPaths
from utils.config import COLMAP_EXE


def run(cmd, log, label):
    cmd = [str(c) for c in cmd]
    log.info(f"[RUN:{label}] {' '.join(cmd)}")

    proc = subprocess.run(
        cmd,
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

    log = get_logger("openmvs_export", project_root)
    log.info("Starting OpenMVS export")

    # --------------------------------------------------
    # Ask user for sparse root folder
    # --------------------------------------------------
    sparse_root = Path(
        input("Enter FULL path to sparse output folder: ").strip()
    ).resolve()

    if not sparse_root.exists():
        raise RuntimeError(f"Sparse root not found: {sparse_root}")

    export_json = sparse_root / "export_ready.json"
    if not export_json.exists():
        raise RuntimeError(f"export_ready.json not found in {sparse_root}")

    meta = json.loads(export_json.read_text())
    model_dir = meta.get("model_dir")

    if not model_dir:
        raise RuntimeError("export_ready.json missing 'model_dir'")

    sparse_model = sparse_root / model_dir
    if not sparse_model.exists():
        raise RuntimeError(f"Sparse model folder not found: {sparse_model}")

    required = {"cameras.bin", "images.bin", "points3D.bin"}
    if not required.issubset({f.name for f in sparse_model.iterdir()}):
        raise RuntimeError("Sparse model missing required COLMAP bin files")

    # --------------------------------------------------
    # Prepare undistorted output
    # --------------------------------------------------
    undistorted = paths.openmvs / "undistorted"
    if undistorted.exists() and force:
        shutil.rmtree(undistorted)

    undistorted.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # COLMAP image undistortion (COLMAP format)
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
    )

    # Remove stereo folder if created (OpenMVS requirement)
    stereo = undistorted / "stereo"
    if stereo.exists():
        shutil.rmtree(stereo)

    # Validate undistorted sparse
    undistorted_sparse = undistorted / "sparse"
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
    )

    if not scene.exists() or scene.stat().st_size < 50_000:
        raise RuntimeError("scene.mvs not generated or invalid")

    log.info(f"OpenMVS scene successfully created: {scene}")
