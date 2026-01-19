#!/usr/bin/env python3
"""
openmvs_export.py

MARK-2 OpenMVS Export Stage (Canonical)
--------------------------------------
- Consumes sparse/export_ready.json
- Runs COLMAP image undistorter
- Converts to OpenMVS scene.mvs
- Strictly config-driven tool resolution
"""

import json
import shutil
from pathlib import Path

from utils.paths import ProjectPaths
from tool_runner import ToolRunner, ToolExecutionError


REQUIRED_SPARSE_FILES = {"cameras.bin", "images.bin", "points3D.bin"}


# ============================================================
# Helpers
# ============================================================

def _validate_sparse_model(model_dir: Path) -> None:
    files = {p.name for p in model_dir.iterdir()}
    missing = REQUIRED_SPARSE_FILES - files
    if missing:
        raise RuntimeError(f"[openmvs] Sparse model missing files: {missing}")


def _load_runtime_config(run_root: Path) -> dict:
    snapshot = run_root / "config_snapshot.json"
    if not snapshot.exists():
        raise RuntimeError(
            "[openmvs] No runtime config available (config_snapshot.json missing)"
        )
    return json.loads(snapshot.read_text(encoding="utf-8"))


# ============================================================
# Stage entry
# ============================================================

def run(run_root: Path, project_root: Path, force: bool, logger) -> None:
    run_root = run_root.resolve()
    project_root = project_root.resolve()

    paths = ProjectPaths(project_root)
    paths.ensure_all()
    paths.validate()

    logger.info("[openmvs] Stage started")

    # --------------------------------------------------
    # ToolRunner (authoritative)
    # --------------------------------------------------
    runtime_config = _load_runtime_config(run_root)
    tool_runner = ToolRunner(config=runtime_config, logger=logger)

    # --------------------------------------------------
    # Sparse model contract
    # --------------------------------------------------
    meta_path = paths.sparse / "export_ready.json"
    if not meta_path.exists():
        raise RuntimeError("[openmvs] export_ready.json not found")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    model_dir = paths.sparse / meta.get("model_dir", "")

    if not model_dir.exists():
        raise RuntimeError("[openmvs] Declared sparse model directory missing")

    _validate_sparse_model(model_dir)
    logger.info(f"[openmvs] Using sparse model: {model_dir.name}")

    # --------------------------------------------------
    # Image undistortion (COLMAP)
    # --------------------------------------------------
    undistorted_root = paths.openmvs / "undistorted"
    sparse_out = undistorted_root / "sparse"

    if undistorted_root.exists() and force:
        shutil.rmtree(undistorted_root)

    if not sparse_out.exists():
        undistorted_root.mkdir(parents=True, exist_ok=True)

        try:
            tool_runner.run(
                "colmap",
                [
                    "image_undistorter",
                    "--image_path", str(paths.images_processed),
                    "--input_path", str(model_dir),
                    "--output_path", str(undistorted_root),
                    "--output_type", "COLMAP",
                ],
                cwd=project_root,
            )
        except ToolExecutionError as e:
            raise RuntimeError("[openmvs] COLMAP image_undistorter failed") from e

        if not sparse_out.exists():
            raise RuntimeError("[openmvs] Undistorted sparse output missing")

    # Remove unused stereo directory
    stereo_dir = undistorted_root / "stereo"
    if stereo_dir.exists():
        shutil.rmtree(stereo_dir)

    # --------------------------------------------------
    # OpenMVS conversion (CAPABILITY NAME, NOT BINARY)
    # --------------------------------------------------
    scene_path = paths.openmvs_scene

    if scene_path.exists() and force:
        scene_path.unlink()

    if not scene_path.exists():
        try:
            tool_runner.run(
                "interface",  # ← THIS IS THE FIX
                [
                    "-i", str(undistorted_root),
                    "-o", str(scene_path),
                    "--image-folder", str(undistorted_root / "images"),
                ],
                cwd=project_root,
            )
        except ToolExecutionError as e:
            raise RuntimeError("[openmvs] OpenMVS interface failed") from e

        if not scene_path.exists() or scene_path.stat().st_size == 0:
            raise RuntimeError("[openmvs] scene.mvs not created")

    logger.info("[openmvs] OpenMVS export completed successfully")
