#!/usr/bin/env python3
"""
sparse_reconstruction.py

MARK-2 Sparse Reconstruction Stage (Optimized)
----------------------------------------------
- ToolRunner enforced
- Adaptive mapper strategy (standard vs hierarchical)
- Reduced BA cost for large datasets
- Deterministic model selection
- Resume-safe
"""

import json
import hashlib
import shutil
from pathlib import Path

from utils.paths import ProjectPaths
from config_manager import load_config
from tool_runner import ToolRunner

REQUIRED_FILES = {"cameras.bin", "images.bin", "points3D.bin"}


# --------------------------------------------------
# Hashing
# --------------------------------------------------
def _hash_sparse(model_dir: Path) -> str:
    h = hashlib.sha256()
    for name in sorted(REQUIRED_FILES):
        h.update((model_dir / name).read_bytes())
    return h.hexdigest()


# --------------------------------------------------
# Adaptive Mapper Strategy
# --------------------------------------------------
def _build_mapper_command(paths, num_images: int):
    """
    Choose optimal mapper configuration based on dataset size.
    """

    # Small datasets → standard mapper
    if num_images < 300:
        mode = "mapper"
        extra_args = [
            "--Mapper.num_threads", "-1",
        ]

    # Medium datasets → tuned mapper
    elif num_images < 1000:
        mode = "mapper"
        extra_args = [
            "--Mapper.num_threads", "-1",
            "--Mapper.ba_global_max_num_iterations", "25",
            "--Mapper.ba_local_max_num_iterations", "15",
            "--Mapper.ba_global_images_ratio", "1.3",
            "--Mapper.ba_global_points_ratio", "1.3",
        ]

    # Large datasets → hierarchical mapper
    else:
        mode = "hierarchical_mapper"
        extra_args = [
            "--Mapper.num_threads", "-1",
            "--Mapper.ba_global_max_num_iterations", "20",
            "--Mapper.ba_local_max_num_iterations", "10",
        ]

    cmd = [
        mode,
        "--database_path", paths.database / "database.db",
        "--image_path", paths.images_processed,
        "--output_path", paths.sparse,
    ] + extra_args

    return cmd, mode


# --------------------------------------------------
# Pipeline Stage
# --------------------------------------------------
def run(run_root: Path, project_root: Path, force: bool, logger):
    paths = ProjectPaths(project_root)
    paths.ensure_all()

    logger.info("[sparse] START")

    config = load_config(run_root, logger)
    tool = ToolRunner(config, logger)

    if paths.sparse.exists() and force:
        shutil.rmtree(paths.sparse)

    paths.sparse.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # Dataset Size Detection
    # --------------------------------------------------
    image_files = list(Path(paths.images_processed).glob("*"))
    num_images = len(image_files)

    logger.info(f"[sparse] Images detected: {num_images}")

    cmd, mode = _build_mapper_command(paths, num_images)

    logger.info(f"[sparse] Using COLMAP mode: {mode}")

    # --------------------------------------------------
    # Run COLMAP
    # --------------------------------------------------
    tool.run("colmap", cmd)

    # --------------------------------------------------
    # Deterministic Model Selection
    # --------------------------------------------------
    models = [
        d for d in paths.sparse.iterdir()
        if d.is_dir() and REQUIRED_FILES.issubset({f.name for f in d.iterdir()})
    ]

    if not models:
        raise RuntimeError("[sparse] No valid sparse models produced")

    # Select model with most 3D points (largest points3D.bin)
    best = max(models, key=lambda d: (d / "points3D.bin").stat().st_size)

    for d in models:
        if d != best:
            shutil.rmtree(d)

    meta = {
        "model_dir": best.name,
        "format": "COLMAP",
        "mapper_mode": mode,
        "num_images": num_images,
        "sparse_hash": _hash_sparse(best),
        "ready_for_openmvs": True,
    }

    (paths.sparse / "export_ready.json").write_text(json.dumps(meta, indent=2))

    logger.info("[sparse] COMPLETED")