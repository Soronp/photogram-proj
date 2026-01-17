from pathlib import Path

from utils.paths import ProjectPaths
from config_manager import create_runtime_config, validate_config
from tool_runner import ToolRunner


# --------------------------------------------------
# PIPELINE STAGE
# --------------------------------------------------
def run(run_root: Path, project_root: Path, force: bool, logger):
    paths = ProjectPaths(project_root)
    paths.ensure_all()

    logger.info("[dense] Stage started (OpenMVS)")

    # --------------------------------------------------
    # CONFIG
    # --------------------------------------------------
    config = create_runtime_config(run_root, project_root, logger)
    if not validate_config(config, logger):
        raise RuntimeError("Invalid configuration")

    tool = ToolRunner(config, logger)

    # --------------------------------------------------
    # INPUT VALIDATION
    # --------------------------------------------------
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

    dense_cfg = config["dense_reconstruction"]["openmvs"]

    # --------------------------------------------------
    # DENSIFY POINT CLOUD
    # --------------------------------------------------
    tool.run(
        tool="densify",  # ToolRunner resolves to DensifyPointCloud executable
        args=[
            "-i", str(scene_file),
            "-o", str(fused_file),
            "--working-folder", str(openmvs_root),
            "--resolution-level", str(dense_cfg.get("resolution_level", 1)),
            "--max-resolution", str(dense_cfg.get("max_resolution", 2560)),
            "--min-resolution", str(dense_cfg.get("min_resolution", 640)),
            "--number-views", str(dense_cfg.get("number_views", 8)),
            "--number-views-fuse", str(dense_cfg.get("number_views_fuse", 3)),
            "--estimate-colors", str(dense_cfg.get("estimate_colors", 2)),
            "--estimate-normals", str(dense_cfg.get("estimate_normals", 2)),
            "--filter-point-cloud", str(dense_cfg.get("filter_point_cloud", 1)),
        ],
        cwd=openmvs_root,
    )

    logger.info(f"[dense] Output written: {fused_file}")
    logger.info("[dense] Stage completed")
