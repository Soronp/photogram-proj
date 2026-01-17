#!/usr/bin/env python3
"""
config_manager.py

MARK-2 Configuration Authority (Run-Scoped)
-------------------------------------------
- Creates one immutable config.yaml per run
- Dataset-driven policy adaptation
- Schema + policy merge
- Single source of truth for all pipeline stages
- Deterministic, resume-safe
"""

from pathlib import Path
from copy import deepcopy
import json
import yaml
import hashlib

from utils.paths import ProjectPaths


# ============================================================
# SCHEMA DEFAULTS (MECHANICAL ONLY)
# ============================================================

SCHEMA_DEFAULTS = {
    "project": {
        "name": None,
    },

    "capture": {
        "mode": "expert",
    },

    "tools": {
        "colmap": {
            "executable": "colmap",
        },
        "openmvs": {
            "exporter": "InterfaceCOLMAP",
            "densify": "DensifyPointCloud",
            "mesh": "ReconstructMesh",
            "texture": "TextureMesh",
        },
        "ffmpeg": {
            "executable": "ffmpeg",
        },
    },

    "execution": {
        "use_gpu": True,
        "num_threads": 0,
        "dry_run": False,
    },

    "stages": {
        "sparse": True,
        "dense": True,
        "mesh": True,
        "texture": True,
        "evaluation": True,
    },

    "feature_extraction": {},
    "matching": {},
    "sparse_reconstruction": {},
    "dense_reconstruction": {},
    "mesh": {},
    "texture": {},
    "evaluation": {},
}


# ============================================================
# POLICY BASELINE (INTENTIONAL DEFAULTS)
# ============================================================

BASE_POLICY = {
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
            "dense_reuse_depth": True,
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


# ============================================================
# CONFIG CREATION (RUN-SCOPED)
# ============================================================

def create_runtime_config(run_root: Path, project_root: Path, logger) -> dict:
    """
    Create an immutable, run-scoped config.yaml.
    Refuses to proceed without dataset intelligence.
    """
    run_root = run_root.resolve()
    project_root = project_root.resolve()
    paths = ProjectPaths(run_root)

    config_path = run_root / "config.yaml"
    intel_path = paths.evaluation / "dataset_intelligence.json"

    logger.info("[config] Creating run-specific configuration")

    if config_path.exists():
        logger.info("[config] config.yaml already exists — reusing")
        return load_config(run_root)

    if not intel_path.exists():
        raise FileNotFoundError(
            "dataset_intelligence.json missing — cannot generate config"
        )

    with open(intel_path, "r", encoding="utf-8") as f:
        intel = json.load(f)

    config = deepcopy(SCHEMA_DEFAULTS)
    policy = deepcopy(BASE_POLICY)

    # --------------------------------------------------------
    # Project metadata
    # --------------------------------------------------------

    config["project"]["name"] = project_root.name

    # --------------------------------------------------------
    # Dataset-driven policy adaptation
    # --------------------------------------------------------

    scale = intel["dataset_scale"]
    feat_mp = intel["features"]["mean_per_megapixel"]
    blur_ratio = intel["quality"]["blur"]["low_ratio"]
    orphan_ratio = intel["overlap"]["orphan_ratio"]

    if feat_mp < 300:
        policy["feature_extraction"]["max_num_features"] = 12000
        logger.info("[config] Low texture → increased feature budget")

    if scale in {"large", "massive"}:
        policy["matching"]["method"] = "sequential"
        logger.info("[config] Large dataset → sequential matching")

    if orphan_ratio > 0.30:
        policy["matching"]["method"] = "exhaustive"
        policy["matching"]["max_ratio"] = 0.75
        logger.info("[config] Poor overlap → stricter matching")

    if blur_ratio > 0.4:
        policy["sparse_reconstruction"]["min_num_inliers"] = 20
        policy["matching"]["max_distance"] = 0.65
        logger.info("[config] Blur-heavy → stricter inliers")

    # --------------------------------------------------------
    # Merge policy into schema
    # --------------------------------------------------------

    for section, values in policy.items():
        config[section].update(values)

    # --------------------------------------------------------
    # Persist immutable config
    # --------------------------------------------------------

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    logger.info(f"[config] config.yaml written: {config_path}")
    return config


# ============================================================
# CONFIG LOADING (AUTHORITATIVE)
# ============================================================

def load_config(run_root: Path, logger=None) -> dict:
    """
    Load run-scoped config.yaml.
    No fallback. No defaults. No mutation.
    """
    config_path = Path(run_root).resolve() / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.yaml at {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if logger:
        logger.info("[config] Loaded run configuration")

    return config


# ============================================================
# VALIDATION
# ============================================================

def validate_config(config: dict, logger) -> bool:
    required = {
        "project",
        "tools",
        "execution",
        "feature_extraction",
        "matching",
        "sparse_reconstruction",
        "dense_reconstruction",
    }

    missing = required - config.keys()
    if missing:
        logger.error(f"[config] Missing sections: {missing}")
        return False

    if config["dense_reconstruction"].get("primary") != "OPENMVS":
        logger.error("[config] OPENMVS must be primary dense backend")
        return False

    logger.info("[config] Configuration validation passed")
    return True
