#!/usr/bin/env python3
"""
config_manager.py

MARK-2 Project Configuration Manager (Full-Resolution, Hybrid Dense Aware)
---------------------------------------------------------------------------
- Always generates a fresh project-specific config.yaml
- OpenMVS is primary dense backend, COLMAP is fallback
- Merges defaults with dataset diagnostics
- Deterministic, restart-safe
- Maintains full image resolution (no downsampling)

LOGGER POLICY:
- Logger is injected by runner
- This module NEVER creates its own logger
"""

from pathlib import Path
import yaml
import json
from utils.paths import ProjectPaths


# -------------------------------------------------
# COMPLETE DEFAULT CONFIGURATION (PIPELINE-WIDE)
# -------------------------------------------------

DEFAULT_CONFIG = {
    "project_name": "MARK-2_Project",

    # -----------------
    # Camera calibration
    # -----------------
    "camera": {
        "model": "PINHOLE",
        "single": True
    },

    # -----------------
    # Feature extraction
    # -----------------
    "feature_extraction": {
        "max_num_features": 8192,
        "edge_threshold": 10
    },

    # -----------------
    # Feature matching
    # -----------------
    "matching": {
        "method": "exhaustive",
        "max_num_matches": 32768,
        "max_ratio": 0.8,
        "max_distance": 0.7
    },

    # -----------------
    # Sparse reconstruction
    # -----------------
    "sparse_reconstruction": {
        "method": "GLOMAP",
        "rotation_filtering_angle_threshold": 30,
        "min_num_inliers": 15,
        "min_inlier_ratio": 0.15
    },

    # -----------------
    # Dense reconstruction (HYBRID)
    # -----------------
    "dense_reconstruction": {
        "primary": "OPENMVS",
        "secondary": "COLMAP",

        "openmvs": {
            "resolution_level": 1,   # FULL resolution
            "min_resolution": 640,
            "max_resolution": 2400,
            "num_threads": 0,
            "use_cuda": True,
            "dense_reuse_depth": True
        },

        "colmap": {
            "max_image_size": 2400,
            "patchmatch": {
                "geom_consistency": True,
                "num_iterations": 5,
                "num_samples": 25,
                "cache_size": 32
            }
        }
    },

    # -----------------
    # Mesh generation
    # -----------------
    "mesh": {
        "enabled": True,
        "poisson_depth": 10
    },

    # -----------------
    # Texture mapping
    # -----------------
    "texture": {
        "enabled": True
    },

    # -----------------
    # Evaluation
    # -----------------
    "evaluation": {
        "enabled": True
    }
}


# -------------------------------------------------
# CONFIG CREATION LOGIC (RUN-AWARE)
# -------------------------------------------------

def create_runtime_config(project_root: Path, logger):
    """
    Generate a fresh config.yaml using defaults + dataset diagnostics.

    Args:
        project_root (Path): MARK-2 output root
        logger: Injected run logger (MANDATORY)

    Returns:
        dict: Generated configuration
    """
    project_root = project_root.resolve()
    paths = ProjectPaths(project_root)

    config_path = project_root / "config.yaml"
    diagnostics_path = paths.evaluation / "dataset_diagnostics.json"

    logger.info("[config] Generating runtime configuration")
    logger.info(f"[config] Target path: {config_path}")

    # Deep copy defaults (safe)
    config = yaml.safe_load(yaml.dump(DEFAULT_CONFIG))
    config["project_name"] = project_root.name

    # -----------------------------------------
    # Apply dataset diagnostics if available
    # -----------------------------------------
    if diagnostics_path.exists():
        try:
            with open(diagnostics_path, "r", encoding="utf-8") as f:
                diagnostics = json.load(f)

            logger.info("[config] Dataset diagnostics loaded")

            avg_features = diagnostics.get("avg_features", 4000)
            avg_blur = diagnostics.get("avg_blur", 0.0)
            image_count = diagnostics.get("image_count", 0)

            # ---- Feature extraction tuning ----
            config["feature_extraction"]["max_num_features"] = max(
                2000, int(avg_features * 1.1)
            )

            # ---- Force full resolution ----
            config["dense_reconstruction"]["openmvs"]["resolution_level"] = 1
            config["dense_reconstruction"]["openmvs"]["max_resolution"] = 2400
            config["dense_reconstruction"]["colmap"]["max_image_size"] = 2400

            # ---- Recommendations ----
            recommendations = diagnostics.get("recommendations", [])
            if avg_blur < 0.2:
                recommendations.append(
                    "Average blur is low; consider aggressive filtering"
                )

            if recommendations:
                config["dataset_recommendations"] = recommendations

            logger.info(
                f"[config] Applied diagnostics "
                f"(images={image_count}, avg_features={avg_features})"
            )

        except Exception as exc:
            logger.warning(f"[config] Failed to apply diagnostics: {exc}")

    # -----------------------------------------
    # Write config.yaml
    # -----------------------------------------
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, sort_keys=False)

    logger.info("[config] config.yaml written successfully")
    return config


# -------------------------------------------------
# VALIDATION (LOGGER-INJECTED)
# -------------------------------------------------

def validate_config(config: dict, logger) -> bool:
    required_sections = [
        "camera",
        "feature_extraction",
        "matching",
        "sparse_reconstruction",
        "dense_reconstruction",
    ]

    for section in required_sections:
        if section not in config:
            logger.error(f"[config] Missing section: {section}")
            return False

    dense = config["dense_reconstruction"]
    if dense.get("primary") not in {"OPENMVS", "COLMAP"}:
        logger.error("[config] Invalid dense reconstruction primary backend")
        return False

    logger.info("[config] Configuration validation passed")
    return True
