#!/usr/bin/env python3
"""
openmvs_export.py

MARK-2 OpenMVS Export (Contract-Driven)
--------------------------------------
- Reads export_ready.json
- Uses exactly one sparse model
- Produces openmvs/scene.mvs deterministically
"""

import json
import shutil
import subprocess
from pathlib import Path

from utils.paths import ProjectPaths
from utils.config import COLMAP_EXE

REQUIRED = {"cameras.bin", "images.bin", "points3D.bin"}


def run_command(cmd, label: str, cwd: Path, logger):
    cmd = [str(c) for c in cmd]
    logger.info(f"[openmvs:{label}] {' '.join(cmd)}")

    proc = subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    logger.info(proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(f"{label} failed")


# --------------------------------------------------
# PIPELINE STAGE
# --------------------------------------------------
def run(project_root: Path, force: bool, logger):
    project_root = Path(project_root).resolve()
    paths = ProjectPaths(project_root)
    paths.ensure_all()
    paths.validate()

    logger.info("[openmvs] Starting export")

    meta_path = paths.sparse / "export_ready.json"
    if not meta_path.exists():
        raise RuntimeError("export_ready.json missing")

    meta = json.loads(meta_path.read_text())
    model_dir = paths.sparse / meta["model_dir"]

    if not model_dir.exists():
        raise RuntimeError("Declared sparse model missing")

    if not REQUIRED.issubset({f.name for f in model_dir.iterdir()}):
        raise RuntimeError("Sparse model invalid")

    logger.info(f"[openmvs] Using sparse model: {model_dir.name}")

    undistorted = paths.openmvs / "undistorted"
    if undistorted.exists() and force:
        shutil.rmtree(undistorted)
    undistorted.mkdir(parents=True, exist_ok=True)

    run_command(
        [
            COLMAP_EXE, "image_undistorter",
            "--image_path", paths.images_processed,
            "--input_path", model_dir,
            "--output_path", undistorted,
            "--output_type", "COLMAP",
        ],
        "image_undistorter",
        cwd=project_root,
        logger=logger,
    )

    stereo = undistorted / "stereo"
    if stereo.exists():
        shutil.rmtree(stereo)

    sparse_out = undistorted / "sparse"
    if not sparse_out.exists():
        raise RuntimeError("Undistorted sparse missing")

    scene = paths.openmvs_scene
    if scene.exists() and force:
        scene.unlink()

    run_command(
        [
            "InterfaceCOLMAP",
            "-i", undistorted,
            "-o", scene,
            "--image-folder", undistorted / "images",
        ],
        "InterfaceCOLMAP",
        cwd=project_root,
        logger=logger,
    )

    if not scene.exists() or scene.stat().st_size < 50_000:
        raise RuntimeError("scene.mvs invalid")

    logger.info(f"[openmvs] scene.mvs created successfully")
