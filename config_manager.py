#!/usr/bin/env python3
"""
config_manager.py

MARK-2 Project Configuration Manager
------------------------------------
- Creates project-specific config.yaml if missing
- Ensures all required parameters exist for all pipeline stages
- Supports per-project customization
"""

from pathlib import Path
import yaml
from utils.logger import get_logger

# -------------------------------
# Complete default configuration for ALL pipeline stages
# -------------------------------
DEFAULT_CONFIG = {
    "project_name": "MARK-2_Project",
    
    # Camera calibration (REQUIRED by database_builder.py)
    "camera": {
        "model": "PINHOLE",
        "single": True
    },
    
    # Feature extraction (used by database_builder.py)
    "feature_extraction": {
        "max_num_features": 8192,
        "edge_threshold": 10
    },
    
    # Feature matching (used by matcher.py)
    "matching": {
        "method": "exhaustive",
        "max_num_matches": 32768,
        "max_ratio": 0.8,
        "max_distance": 0.7
    },
    
    # Sparse reconstruction (GLOMAP - used by sparse_reconstruction.py)
    "sparse_reconstruction": {
        "method": "GLOMAP",  # MUST match sparse_reconstruction.py
        "rotation_filtering_angle_threshold": 30,
        "min_num_inliers": 15,
        "min_inlier_ratio": 0.15
    },
    
    # Dense reconstruction
    "dense_reconstruction": {
        "method": "COLMAP"  # Actually used by dense_reconstruction.py
    },
    
    # Mesh generation
    "mesh": {
        "enabled": True,
        "poisson_depth": 10
    },
    
    # Texture mapping
    "texture": {
        "enabled": True
    },
    
    # Evaluation
    "evaluation": {
        "enabled": True
    }
}

# -------------------------------
# Config Manager
# -------------------------------
def create_or_update_config(project_root: Path, overrides: dict = None):
    """
    Ensure a valid config.yaml exists in project_root.
    """
    logger = get_logger("config_manager", project_root)
    config_path = project_root / "config.yaml"
    
    logger.info(f"Managing configuration at: {config_path}")
    
    # Start with defaults
    config = DEFAULT_CONFIG.copy()
    
    # Update project name from directory
    config["project_name"] = project_root.name
    
    # Apply overrides if provided
    if overrides:
        logger.info("Applying configuration overrides")
        # Deep merge dictionaries
        def deep_update(target, source):
            for key, value in source.items():
                if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                    deep_update(target[key], value)
                else:
                    target[key] = value
        
        deep_update(config, overrides)
    
    # Check if config already exists
    existing_config = None
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                existing_config = yaml.safe_load(f) or {}
            logger.info("Existing config.yaml found")
            
            # Validate existing config has all required sections
            needs_update = False
            for section, defaults in DEFAULT_CONFIG.items():
                if section not in existing_config:
                    existing_config[section] = defaults
                    needs_update = True
                    logger.warning(f"Added missing section: {section}")
            
            if needs_update:
                with open(config_path, "w") as f:
                    yaml.dump(existing_config, f, default_flow_style=False)
                logger.info("Updated existing config.yaml with missing sections")
            
            return existing_config
        except yaml.YAMLError as e:
            logger.error(f"Invalid existing config.yaml: {e}")
            # Fall through to create new
    
    # Create new config.yaml
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"Created new config.yaml at {config_path}")
    
    return config

# -------------------------------
# Validation function (optional but recommended)
# -------------------------------
def validate_config(config: dict, logger) -> bool:
    """Validate config has all required parameters."""
    required_sections = ["camera", "matching", "sparse_reconstruction"]
    
    for section in required_sections:
        if section not in config:
            logger.error(f"Missing required config section: {section}")
            return False
    
    # Specific validations
    if "camera" in config:
        if "model" not in config["camera"]:
            logger.error("Camera model not specified in config")
            return False
    
    if "sparse_reconstruction" in config:
        if config["sparse_reconstruction"].get("method") != "GLOMAP":
            logger.warning("Sparse reconstruction method should be 'GLOMAP' for your pipeline")
    
    return True

# -------------------------------
# CLI Interface
# -------------------------------
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="MARK-2 Config Manager")
    parser.add_argument("project_root", type=Path, help="Project root directory")
    parser.add_argument("--override", type=str, help="YAML file with config overrides")
    args = parser.parse_args()
    
    overrides = {}
    if args.override:
        override_path = Path(args.override)
        if override_path.exists():
            with open(override_path) as f:
                overrides = yaml.safe_load(f) or {}
    
    config = create_or_update_config(args.project_root, overrides)
    
    # Validate
    logger = get_logger("config_manager", args.project_root)
    if validate_config(config, logger):
        logger.info("Configuration is valid")
    else:
        logger.error("Configuration validation failed")


if __name__ == "__main__":
    main()