#!/usr/bin/env python3
"""
config_manager.py

MARK-2 Project Configuration Manager (Hybrid Dense Aware)
---------------------------------------------------------
- Always generates a fresh project-specific config.yaml
- OpenMVS is primary dense backend, COLMAP is fallback
- Merges defaults with dataset diagnostics
- Deterministic, restart-safe
"""

from pathlib import Path
import yaml
import json
from utils.logger import get_logger

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

        # ---- OpenMVS parameters ----
        "openmvs": {
            "resolution_level": 1,
            "min_resolution": 640,
            "max_resolution": 2400,
            "num_threads": 0,        # 0 = auto
            "use_cuda": True,
            "dense_reuse_depth": True
        },

        # ---- COLMAP fallback parameters ----
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
# CONFIG CREATION LOGIC
# -------------------------------------------------
def create_runtime_config(project_root: Path):
    """
    Generate a fresh config.yaml using defaults + dataset diagnostics.
    """
    project_root = project_root.resolve()
    logger = get_logger("config_manager", project_root)

    config_path = project_root / "config.yaml"
    diagnostics_path = project_root / "evaluation" / "dataset_diagnostics.json"

    logger.info(f"Generating runtime configuration: {config_path}")

    # Deep copy defaults
    config = yaml.safe_load(yaml.dump(DEFAULT_CONFIG))
    config["project_name"] = project_root.name

    # -----------------------------------------
    # Apply dataset diagnostics if available
    # -----------------------------------------
    if diagnostics_path.exists():
        try:
            with open(diagnostics_path, "r", encoding="utf-8") as f:
                diagnostics = json.load(f)

            logger.info("Dataset diagnostics loaded")

            preprocessing = diagnostics.get("preprocessing", {})
            downsample = preprocessing.get("downsample_factor", 1.0)

            avg_features = diagnostics.get("avg_features", 4000)
            avg_blur = diagnostics.get("avg_blur", 0.0)
            image_count = diagnostics.get("image_count", 0)

            # ---- Feature extraction tuning ----
            config["feature_extraction"]["max_num_features"] = max(
                2000, int(avg_features * 1.1)
            )

            # ---- OpenMVS resolution tuning ----
            openmvs_cfg = config["dense_reconstruction"]["openmvs"]

            if image_count <= 50:
                openmvs_cfg["resolution_level"] = 0
            elif image_count <= 150:
                openmvs_cfg["resolution_level"] = 1
            else:
                openmvs_cfg["resolution_level"] = 2

            openmvs_cfg["max_resolution"] = int(
                openmvs_cfg["max_resolution"] * downsample
            )

            # ---- COLMAP fallback tuning ----
            colmap_cfg = config["dense_reconstruction"]["colmap"]
            colmap_cfg["max_image_size"] = int(
                colmap_cfg["max_image_size"] * downsample
            )

            # ---- Blur-based recommendations ----
            recommendations = diagnostics.get("recommendations", [])
            if avg_blur < 0.2:
                recommendations.append(
                    "Average blur is low; aggressive filtering may be required"
                )

            if recommendations:
                config["dataset_recommendations"] = recommendations

            logger.info(
                f"Applied diagnostics overrides "
                f"(downsample={downsample}, images={image_count})"
            )

        except Exception as e:
            logger.warning(f"Failed to apply diagnostics: {e}")

    # -----------------------------------------
    # Write config.yaml
    # -----------------------------------------
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, sort_keys=False)

    logger.info("Runtime config.yaml written successfully")
    return config

# -------------------------------------------------
# VALIDATION
# -------------------------------------------------
def validate_config(config: dict, logger) -> bool:
    required_sections = [
        "camera",
        "feature_extraction",
        "matching",
        "sparse_reconstruction",
        "dense_reconstruction"
    ]

    for section in required_sections:
        if section not in config:
            logger.error(f"Missing config section: {section}")
            return False

    dense = config["dense_reconstruction"]
    if dense.get("primary") not in {"OPENMVS", "COLMAP"}:
        logger.error("Invalid dense reconstruction primary backend")
        return False

    return True

# -------------------------------------------------
# CLI
# -------------------------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(description="MARK-2 Runtime Config Manager")
    parser.add_argument("project_root", type=Path)
    args = parser.parse_args()

    logger = get_logger("config_manager", args.project_root)
    config = create_runtime_config(args.project_root)

    if validate_config(config, logger):
        logger.info("Configuration validated and ready")
    else:
        logger.error("Configuration validation failed")

if __name__ == "__main__":
    main()
