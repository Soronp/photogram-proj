#!/usr/bin/env python3
"""
dense_reconstruction.py

MARK-2 Dense Reconstruction Stage (OpenMVS)
------------------------------------------
- ToolRunner enforced
- GPU opt-in via config
- Resume-safe
"""

from pathlib import Path

from utils.paths import ProjectPaths
from config_manager import load_config
from tool_runner import ToolRunner


def run(run_root: Path, project_root: Path, force: bool, logger):
    paths = ProjectPaths(project_root)
    paths.ensure_all()

    logger.info("[dense] START")

    config = load_config(run_root, logger)
    tool = ToolRunner(config, logger)

    openmvs_root = paths.openmvs
    scene = openmvs_root / "scene.mvs"

    if not scene.exists():
        raise FileNotFoundError("scene.mvs missing")

    paths.dense.mkdir(parents=True, exist_ok=True)
    fused = paths.dense / "fused.ply"

    if fused.exists() and not force:
        logger.info("[dense] Output exists — skipping")
        return

    cfg = config["dense_reconstruction"]["openmvs"]

    tool.run(
        "densify",
        [
            "-i", scene,
            "-o", fused,
            "--working-folder", openmvs_root,
            "--resolution-level", cfg.get("resolution_level", 1),
            "--min-resolution", cfg.get("min_resolution", 640),
            "--max-resolution", cfg.get("max_resolution", 2400),
            "--number-views", cfg.get("number_views", 8),
            "--number-views-fuse", cfg.get("number_views_fuse", 3),
            "--estimate-colors", cfg.get("estimate_colors", 2),
            "--estimate-normals", cfg.get("estimate_normals", 2),
            "--filter-point-cloud", cfg.get("filter_point_cloud", 1),
        ],
        cwd=openmvs_root,
    )

    logger.info("[dense] COMPLETED")
