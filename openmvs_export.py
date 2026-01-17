#!/usr/bin/env python3
"""
openmvs_export.py

MARK-2 OpenMVS Export Stage (Rewritten)
--------------------------------------
Responsibilities:
- Consume export_ready.json
- Validate exactly one sparse model
- Run COLMAP image undistorter safely
- Produce deterministic openmvs/scene.mvs
- Validate all outputs and log clearly
"""

import json
import shutil
from pathlib import Path
from utils.paths import ProjectPaths
from utils.config import COLMAP_EXE, OPENMVS_TOOLS
from tool_runner import ToolRunner, ToolExecutionError

REQUIRED_FILES = {"cameras.bin", "images.bin", "points3D.bin"}


# --------------------------------------------------
# Helper: safe command runner
# --------------------------------------------------
def run_command(tool_runner: ToolRunner, tool_name: str, args: list, cwd: Path, logger, output_check: Path | None = None):
    """
    Run a tool using ToolRunner and optionally verify output.

    Args:
        tool_runner: instance of ToolRunner
        tool_name: logical tool name (e.g., 'colmap', 'interface_colmap')
        args: command-line arguments
        cwd: working directory
        logger: logging object
        output_check: path to verify existence after run
    """
    try:
        proc = tool_runner.run(tool_name, args, cwd=cwd)
    except ToolExecutionError:
        logger.error(f"[openmvs:{tool_name}] Execution failed")
        raise

    if output_check:
        if not output_check.exists() or (output_check.is_file() and output_check.stat().st_size == 0):
            raise RuntimeError(f"[openmvs:{tool_name}] Expected output missing or empty: {output_check}")


# --------------------------------------------------
# Pipeline Stage
# --------------------------------------------------
def run(run_root: Path, project_root: Path, force: bool, logger):
    project_root = project_root.resolve()
    paths = ProjectPaths(project_root)
    paths.ensure_all()
    paths.validate()

    logger.info("[openmvs] Starting OpenMVS export stage")

    # Initialize ToolRunner with runtime config
    # It uses config defaults and PATH-resolved executables
    tool_runner = ToolRunner(config={}, logger=logger)

    # --------------------------------------------------
    # Contract: export_ready.json
    # --------------------------------------------------
    meta_path = paths.sparse / "export_ready.json"
    if not meta_path.exists():
        raise RuntimeError("[openmvs] export_ready.json missing")

    meta = json.loads(meta_path.read_text())
    model_dir = paths.sparse / meta.get("model_dir", "")

    if not model_dir.exists():
        raise RuntimeError("[openmvs] Declared sparse model does not exist")

    model_files = {f.name for f in model_dir.iterdir()}
    if not REQUIRED_FILES.issubset(model_files):
        raise RuntimeError("[openmvs] Sparse model incomplete or invalid")

    logger.info(f"[openmvs] Using sparse model: {model_dir.name}")

    # --------------------------------------------------
    # Undistortion
    # --------------------------------------------------
    undistorted = paths.openmvs / "undistorted"
    if undistorted.exists() and force:
        shutil.rmtree(undistorted)
    undistorted.mkdir(parents=True, exist_ok=True)

    sparse_out = undistorted / "sparse"

    run_command(
        tool_runner,
        "colmap",
        [
            "image_undistorter",
            "--image_path", str(paths.images_processed),
            "--input_path", str(model_dir),
            "--output_path", str(undistorted),
            "--output_type", "COLMAP",
        ],
        cwd=project_root,
        logger=logger,
        output_check=sparse_out
    )

    # COLMAP creates stereo/ by default — remove for OpenMVS parity
    stereo_dir = undistorted / "stereo"
    if stereo_dir.exists():
        shutil.rmtree(stereo_dir)

    # --------------------------------------------------
    # OpenMVS Interface
    # --------------------------------------------------
    scene = paths.openmvs_scene
    if scene.exists() and force:
        scene.unlink()

    run_command(
        tool_runner,
        "interface_colmap",
        [
            "-i", str(undistorted),
            "-o", str(scene),
            "--image-folder", str(undistorted / "images"),
        ],
        cwd=project_root,
        logger=logger,
        output_check=scene
    )

    logger.info("[openmvs] scene.mvs created successfully")
