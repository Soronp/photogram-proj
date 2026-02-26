#!/usr/bin/env python3
"""
dense_reconstruction.py

MARK-2 Dense Reconstruction Stage (OpenMVS - Adaptive)
------------------------------------------------------
- ToolRunner enforced
- Resume-safe
- Uses sparse evaluation metrics to adapt OpenMVS parameters
- Deterministic behavior
"""

import json
from pathlib import Path

from utils.paths import ProjectPaths
from config_manager import load_config
from tool_runner import ToolRunner


# --------------------------------------------------
# Adaptive Parameter Logic
# --------------------------------------------------
def compute_adaptive_params(sparse_metrics: dict, base_cfg: dict) -> dict:
    """
    Deterministically adapt OpenMVS parameters based on sparse quality.
    """

    cfg = dict(base_cfg)

    reproj = sparse_metrics.get("mean_reprojection_error") or 1.0
    images = sparse_metrics.get("num_images") or 0
    points = sparse_metrics.get("num_points") or 0
    mean_track = sparse_metrics.get("mean_track_length") or 2.0
    mean_obs = sparse_metrics.get("mean_observations_per_image") or 50.0

    # --------------------------------------------------
    # Dataset size scaling
    # --------------------------------------------------
    if images > 800:
        cfg["resolution_level"] = 2
    elif images > 300:
        cfg["resolution_level"] = 1
    else:
        cfg["resolution_level"] = 0

    # --------------------------------------------------
    # Reprojection error sensitivity
    # --------------------------------------------------
    if reproj > 1.2:
        cfg["number_views"] = 12
        cfg["number_views_fuse"] = 4
        cfg["filter_point_cloud"] = 2
    elif reproj > 0.8:
        cfg["number_views"] = 10
        cfg["number_views_fuse"] = 3
        cfg["filter_point_cloud"] = 1
    else:
        cfg["number_views"] = 8
        cfg["number_views_fuse"] = 3
        cfg["filter_point_cloud"] = 1

    # --------------------------------------------------
    # Weak sparse safeguard
    # --------------------------------------------------
    if points < 20000 or mean_track < 2.5:
        cfg["filter_point_cloud"] = 2
        cfg["number_views"] = max(cfg["number_views"], 12)

    # --------------------------------------------------
    # High quality dataset optimization
    # --------------------------------------------------
    if reproj < 0.5 and mean_track > 4.0:
        cfg["resolution_level"] = 0
        cfg["number_views"] = 6
        cfg["number_views_fuse"] = 2

    return cfg


# --------------------------------------------------
# Pipeline Stage
# --------------------------------------------------
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

    # --------------------------------------------------
    # Load sparse evaluation metrics
    # --------------------------------------------------
    metrics_path = paths.evaluation / "sparse_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(
            "Sparse metrics not found. Run sparse_evaluation first."
        )

    sparse_metrics = json.loads(metrics_path.read_text())

    base_cfg = config["dense_reconstruction"]["openmvs"]
    cfg = compute_adaptive_params(sparse_metrics, base_cfg)

    logger.info("[dense] Adaptive OpenMVS configuration:")
    for k, v in cfg.items():
        logger.info(f"  {k}: {v}")

    # --------------------------------------------------
    # Run OpenMVS densify
    # --------------------------------------------------
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