#!/usr/bin/env python3
"""
sparse_reconstruction.py

MARK-2 Sparse Reconstruction Stage (GPU Optimized)
--------------------------------------------------
- GPU-accelerated feature extraction and matching
- Adaptive mapper strategy (standard vs hierarchical)
- Deterministic model selection (largest points3D.bin)
- Resumable and safe; old models preserved
- Export-ready JSON for downstream stages
"""

import json
import hashlib
from pathlib import Path
from typing import Tuple

from utils.paths import ProjectPaths
from config_manager import load_config
from tool_runner import ToolRunner, ToolExecutionError

REQUIRED_FILES = {"cameras.bin", "images.bin", "points3D.bin"}


# --------------------------------------------------
# Hashing for sparse models
# --------------------------------------------------
def _hash_sparse(model_dir: Path) -> str:
    """Compute a deterministic hash of all required sparse model files."""
    h = hashlib.sha256()
    for name in sorted(REQUIRED_FILES):
        h.update((model_dir / name).read_bytes())
    return h.hexdigest()


# --------------------------------------------------
# Adaptive Mapper Strategy
# --------------------------------------------------
def _build_mapper_command(paths: ProjectPaths, num_images: int, mode_override: str = None) -> Tuple[list, str]:
    """
    Build COLMAP mapper command adaptively based on dataset size.
    Returns the command list and the chosen mode.
    """
    # Manual override (fallback)
    if mode_override:
        mode = mode_override
        extra_args = ["--Mapper.num_threads", "-1"]
    else:
        # Small datasets → standard mapper
        if num_images < 300:
            mode = "mapper"
            extra_args = ["--Mapper.num_threads", "-1"]
        # Medium datasets → tuned standard mapper
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
        "--database_path", str(paths.database / "database.db"),
        "--image_path", str(paths.images_processed),
        "--output_path", str(paths.sparse),
    ] + extra_args

    return cmd, mode


# --------------------------------------------------
# GPU Feature Extraction
# --------------------------------------------------
def _extract_features_gpu(tool: ToolRunner, paths: ProjectPaths, num_images: int):
    """Extract SIFT features using GPU with optional adaptive tuning."""
    cmd = [
        "feature_extractor",
        "--database_path", str(paths.database / "database.db"),
        "--image_path", str(paths.images_processed),
        "--FeatureExtraction.use_gpu", "1",
        "--SiftExtraction.max_num_features", "8192",
        "--SiftExtraction.estimate_affine_shape", "1",
        "--SiftExtraction.domain_size_pooling", "1",
    ]

    # Adaptive SIFT tuning for very large datasets
    if num_images > 1000:
        cmd += [
            "--SiftExtraction.num_octaves", "5",
            "--SiftExtraction.octave_resolution", "4",
        ]

    tool.run("colmap", cmd)


# --------------------------------------------------
# GPU Feature Matching
# --------------------------------------------------
def _match_features_gpu(tool: ToolRunner, paths: ProjectPaths, num_images: int):
    """
    Match features using GPU. Adapts matcher type based on dataset size.
    COLMAP 3.13 syntax: remove unrecognized options like '--GuidedMatching'.
    """
    matcher = "exhaustive_matcher" if num_images < 1000 else "sequential_matcher"

    cmd = [
        matcher,
        "--database_path", str(paths.database / "database.db"),
        "--FeatureMatching.use_gpu", "1",
    ]

    tool.run("colmap", cmd)


# --------------------------------------------------
# Pipeline Stage
# --------------------------------------------------
def run(run_root: Path, project_root: Path, force: bool, logger):
    paths = ProjectPaths(project_root)
    paths.ensure_all()

    logger.info("[sparse] START")
    config = load_config(run_root, logger)
    tool = ToolRunner(config, logger)

    # --------------------------------------------------
    # Detect number of images
    # --------------------------------------------------
    image_files = list(Path(paths.images_processed).glob("*"))
    num_images = len(image_files)
    logger.info(f"[sparse] Images detected: {num_images}")

    # --------------------------------------------------
    # GPU Feature Extraction
    # --------------------------------------------------
    logger.info("[sparse] Extracting features (GPU)...")
    _extract_features_gpu(tool, paths, num_images)

    # --------------------------------------------------
    # GPU Feature Matching
    # --------------------------------------------------
    logger.info("[sparse] Matching features (GPU)...")
    _match_features_gpu(tool, paths, num_images)

    # --------------------------------------------------
    # Run COLMAP Mapper with fallback for large datasets
    # --------------------------------------------------
    try:
        cmd, mode = _build_mapper_command(paths, num_images)
        logger.info(f"[sparse] Using COLMAP mode: {mode}")
        tool.run("colmap", cmd)
    except ToolExecutionError:
        if num_images >= 1000:
            logger.warning("[sparse] Hierarchical mapper failed; falling back to standard mapper...")
            cmd, mode = _build_mapper_command(paths, num_images, mode_override="mapper")
            tool.run("colmap", cmd)
        else:
            raise

    # --------------------------------------------------
    # Deterministic model selection (largest points3D.bin)
    # --------------------------------------------------
    models = [
        d for d in paths.sparse.iterdir()
        if d.is_dir() and REQUIRED_FILES.issubset({f.name for f in d.iterdir()})
    ]
    if not models:
        raise RuntimeError("[sparse] No valid sparse models produced")

    best = max(models, key=lambda d: (d / "points3D.bin").stat().st_size)

    # Note: Old models are preserved; only export_ready.json points to the best
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