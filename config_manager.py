#!/usr/bin/env python3
"""
config_manager.py

MARK-2 Adaptive Configuration Manager
-------------------------------------
- Generates a run-specific config.yaml
- Fully driven by dataset_intelligence.json
- Deterministic, explainable, restart-safe
- No implicit defaults, no silent heuristics

LOGGER POLICY:
- Logger is injected by runner
- This module NEVER creates its own logger
"""

from pathlib import Path
import yaml
import json
from copy import deepcopy
from utils.paths import ProjectPaths


# =================================================
# BASELINE CONFIG (ONLY WHAT CAN BE MODIFIED)
# =================================================

BASE_CONFIG = {
    "camera": {
        "model": "PINHOLE",
        "single": True,
    },

    "feature_extraction": {
        "max_num_features": 8192,
        "edge_threshold": 10,
    },

    "matching": {
        "method": "exhaustive",
        "max_num_matches": 32768,
        "max_ratio": 0.8,
        "max_distance": 0.7,
    },

    "sparse_reconstruction": {
        "method": "GLOMAP",
        "rotation_filtering_angle_threshold": 30,
        "min_num_inliers": 15,
        "min_inlier_ratio": 0.15,
    },

    "dense_reconstruction": {
        "primary": "OPENMVS",

        "openmvs": {
            "resolution_level": 1,
            "min_resolution": 640,
            "max_resolution": 2400,
            "num_threads": 0,
            "use_cuda": True,
            "dense_reuse_depth": True,
        },

        "colmap_fallback": {
            "max_image_size": 2400,
        },
    },

    "mesh": {
        "enabled": True,
        "poisson_depth": 10,
    },

    "texture": {
        "enabled": True,
    },

    "evaluation": {
        "enabled": True,
    },
}


# =================================================
# POLICY ENGINE
# =================================================

def create_runtime_config(run_root: Path, project_root: Path, logger):
    """
    Generate a dataset-precise config.yaml for the current run.
    """
    run_root = run_root.resolve()
    project_root = project_root.resolve()
    paths = ProjectPaths(run_root)



    config_path = project_root / "config.yaml"
    intelligence_path = paths.evaluation / "dataset_intelligence.json"

    logger.info("[config] MARK-2 adaptive config generation started")
    logger.info(f"[config] Project root: {project_root}")

    config = deepcopy(BASE_CONFIG)
    config["project_name"] = project_root.name

    if not intelligence_path.exists():
        logger.error("[config] dataset_intelligence.json not found — refusing to guess")
        raise FileNotFoundError(intelligence_path)

    with open(intelligence_path, "r", encoding="utf-8") as f:
        intel = json.load(f)

    logger.info("[config] Dataset intelligence loaded")

    # =================================================
    # Unpack intelligence (explicitly)
    # =================================================

    img_count = intel["image_count"]
    scale = intel["dataset_scale"]

    blur_mean = intel["quality"]["blur"]["mean"]
    blur_low_ratio = intel["quality"]["blur"]["low_ratio"]

    feat_mp = intel["features"]["mean_per_megapixel"]
    low_feat_ratio = intel["features"]["low_density_ratio"]

    orphan_ratio = intel["overlap"]["orphan_ratio"]
    mean_hamming = intel["overlap"]["mean_hamming_distance"]

    aspect_std = intel["viewpoint"]["aspect_ratio_std"]

    logger.info(
        "[config] Dataset summary | "
        f"images={img_count}, scale={scale}, "
        f"blur_mean={blur_mean:.1f}, "
        f"feat/mp={feat_mp:.0f}, "
        f"orphan_ratio={orphan_ratio:.2f}"
    )

    # =================================================
    # FEATURE EXTRACTION POLICY
    # =================================================

    if feat_mp < 300:
        config["feature_extraction"]["max_num_features"] = 12000
        logger.info("[config] Low texture → increasing feature budget")

    elif feat_mp > 1200:
        config["feature_extraction"]["max_num_features"] = 6000
        logger.info("[config] High texture → reducing redundant features")

    # =================================================
    # MATCHING POLICY (CRITICAL)
    # =================================================

    if scale in {"large", "massive"}:
        config["matching"]["method"] = "sequential"
        logger.info("[config] Large dataset → sequential matching enforced")

    if orphan_ratio > 0.30 or (mean_hamming and mean_hamming > 22):
        config["matching"]["method"] = "exhaustive"
        config["matching"]["max_ratio"] = 0.75
        logger.info("[config] Poor overlap detected → exhaustive + stricter ratio")

    if blur_low_ratio > 0.4:
        config["matching"]["max_distance"] = 0.65
        logger.info("[config] High blur ratio → tightening descriptor distance")

    # =================================================
    # SPARSE RECONSTRUCTION POLICY
    # =================================================

    if blur_low_ratio > 0.4:
        config["sparse_reconstruction"]["min_num_inliers"] = 20
        config["sparse_reconstruction"]["min_inlier_ratio"] = 0.2
        logger.info("[config] Blur-heavy dataset → stricter inlier thresholds")

    if aspect_std < 0.05:
        config["sparse_reconstruction"]["rotation_filtering_angle_threshold"] = 20
        logger.info("[config] Low viewpoint diversity → stricter rotation filtering")

    # =================================================
    # DENSE RECONSTRUCTION POLICY
    # =================================================

    # MARK-2 rule: never downsample globally
    config["dense_reconstruction"]["openmvs"]["resolution_level"] = 1

    if scale in {"large", "massive"}:
        config["dense_reconstruction"]["openmvs"]["dense_reuse_depth"] = True
        logger.info("[config] Large dataset → depth reuse enabled")

    # =================================================
    # FAIL-SAFE FLAGS (NON-SILENT)
    # =================================================

    flags = intel.get("flags", [])
    if flags:
        config["dataset_flags"] = flags
        logger.warning(f"[config] Dataset flags propagated: {flags}")

    # =================================================
    # WRITE CONFIG
    # =================================================

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, sort_keys=False)

    logger.info(f"[config] config.yaml written: {config_path}")
    logger.info("[config] MARK-2 adaptive config generation completed")

    return config


# =================================================
# VALIDATION
# =================================================

def validate_config(config: dict, logger) -> bool:
    required = {
        "camera",
        "feature_extraction",
        "matching",
        "sparse_reconstruction",
        "dense_reconstruction",
    }

    missing = required - config.keys()
    if missing:
        logger.error(f"[config] Missing required sections: {missing}")
        return False

    if config["dense_reconstruction"]["primary"] != "OPENMVS":
        logger.error("[config] MARK-2 requires OPENMVS as primary dense backend")
        return False

    logger.info("[config] Configuration validation passed")
    return True
