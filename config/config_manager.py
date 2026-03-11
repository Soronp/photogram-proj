#!/usr/bin/env python3
"""
config_manager.py

Runtime Configuration Manager
"""

from pathlib import Path
from copy import deepcopy
import yaml
import json


# ==========================================================
# DEFAULT CONFIGURATION
# ==========================================================

SCHEMA_DEFAULTS = {

    "project": {
        "name": None,
        "dataset_path": None
    },

    "execution": {
        "use_gpu": True,
        "num_threads": 0,
        "dry_run": False
    },

    "tools": {

        "colmap": "colmap",
        "glomap": "glomap",
        "ffmpeg": "ffmpeg",

        "openmvs": {
            "interface": "InterfaceCOLMAP",
            "densifypointcloud": "DensifyPointCloud",
            "reconstructmesh": "ReconstructMesh",
            "refinemesh": "RefineMesh",        # ← FIXED
            "texturemesh": "TextureMesh"
        }
    },

    "feature_extraction": {
        "max_num_features": 8192,
        "edge_threshold": 10
    },

    "matching": {
        "method": "exhaustive",
        "max_num_matches": 32768,
        "max_ratio": 0.8,
        "max_distance": 0.7
    },

    "sparse_reconstruction": {
        "method": "GLOMAP",
        "min_num_inliers": 15,
        "min_inlier_ratio": 0.15
    },

    "dense_reconstruction": {

        "primary": "OPENMVS",

        "openmvs": {
            "resolution_level": 1,
            "min_resolution": 640,
            "max_resolution": 2400,
            "reuse_depth_maps": True
        }
    },

    "mesh": {
        "enabled": True,
        "poisson_depth": 10
    },

    "texture": {
        "enabled": True,
        "max_texture_size": 4096
    },

    "evaluation": {
        "enabled": True
    }
}


# ==========================================================
# CREATE CONFIG
# ==========================================================

def create_runtime_config(run_root: Path, dataset_path: Path, logger):

    run_root = Path(run_root).resolve()
    dataset_path = Path(dataset_path).resolve()

    config_yaml = run_root / "config.yaml"
    snapshot = run_root / "config_snapshot.json"

    logger.info("[config] initializing configuration")

    if config_yaml.exists():

        logger.info("[config] existing config found")

        config = load_config(run_root, logger)

        config = upgrade_config(config)

        return config

    config = deepcopy(SCHEMA_DEFAULTS)

    config["project"]["name"] = dataset_path.name
    config["project"]["dataset_path"] = str(dataset_path)

    # ------------------------------------------------------
    # Dataset scan (fast)
    # ------------------------------------------------------

    image_ext = {".jpg", ".jpeg", ".png"}

    image_count = sum(
        1 for f in dataset_path.glob("*")
        if f.suffix.lower() in image_ext
    )

    logger.info(f"[config] dataset images detected: {image_count}")

    # small dataset tuning
    if image_count < 50:

        config["feature_extraction"]["max_num_features"] = 12000

        logger.info("[config] small dataset → increasing features")

    # large dataset tuning
    elif image_count > 500:

        config["matching"]["method"] = "sequential"

        logger.info("[config] large dataset → sequential matching")

    # ------------------------------------------------------

    with open(config_yaml, "w", encoding="utf-8") as f:

        yaml.safe_dump(
            config,
            f,
            sort_keys=False,
            allow_unicode=True
        )

    with open(snapshot, "w", encoding="utf-8") as f:

        json.dump(config, f, indent=2)

    logger.info("[config] config.yaml written")

    validate_config(config, logger)

    return config


# ==========================================================
# LOAD CONFIG
# ==========================================================

def load_config(run_root: Path, logger=None):

    config_yaml = Path(run_root) / "config.yaml"

    if not config_yaml.exists():
        raise RuntimeError("config.yaml missing")

    config = yaml.safe_load(config_yaml.read_text())

    if logger:
        logger.info("[config] configuration loaded")

    return config


# ==========================================================
# AUTO-UPGRADE OLD CONFIGS
# ==========================================================

def upgrade_config(config):

    updated = deepcopy(SCHEMA_DEFAULTS)

    def merge(dst, src):

        for k, v in src.items():

            if isinstance(v, dict) and k in dst:
                merge(dst[k], v)
            else:
                dst[k] = v

    merge(updated, config)

    return updated


# ==========================================================
# VALIDATION
# ==========================================================

def validate_config(config, logger=None):

    required_sections = {

        "project",
        "execution",
        "tools",
        "feature_extraction",
        "matching",
        "sparse_reconstruction",
        "dense_reconstruction"
    }

    missing = required_sections - config.keys()

    if missing:
        raise ValueError(
            f"config missing required sections: {missing}"
        )

    # Dense backend validation

    if config["dense_reconstruction"]["primary"] != "OPENMVS":

        raise ValueError(
            "OPENMVS must be the primary dense backend"
        )

    # Tool validation

    required_tools = {"colmap", "glomap", "ffmpeg", "openmvs"}

    missing_tools = required_tools - config["tools"].keys()

    if missing_tools:

        raise ValueError(
            f"missing tool definitions: {missing_tools}"
        )

    openmvs_tools = {

        "interface",
        "densifypointcloud",
        "reconstructmesh",
        "texturemesh"
    }

    defined = set(config["tools"]["openmvs"].keys())

    if not openmvs_tools.issubset(defined):

        raise ValueError(
            "OpenMVS tools missing in config"
        )

    if logger:
        logger.info("[config] validation successful")