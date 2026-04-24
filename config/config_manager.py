from copy import deepcopy
from pathlib import Path


# =====================================================
# DEFAULT CONFIG (CONSISTENT + SAFE)
# =====================================================

DEFAULT_CONFIG = {

    "pipeline": {
        "name": "adaptive_multibackend_sfm",
        "mode": "mesh",

        "backends": {
            "sparse": "colmap",

            # 🔥 SAFE DEFAULT (aligned)
            "dense": "nerfstudio",
            "mesh": "auto",          # 🔥 auto-resolve
            "texture": "auto"
        },

        "camera_model": "auto"
    },

    "paths": {
        "project_root": None
    },

    "ingestion": {
        "copy_mode": "copy"
    },

    "downsampling": {
        "enabled": True,
        "target_max_dim": 2400
    },

    "sift": {
        "max_num_features": 16000,
        "max_image_size": 3200,
        "num_threads": -1,
        "use_gpu": True,
        "peak_threshold": 0.004
    },

    "matching": {
        "type": "exhaustive",
        "use_gpu": True
    },

    "sparse": {
        "fallback_to_colmap": True,

        "colmap": {
            "init_min_inliers": 40,
            "abs_min_inliers": 30,
            "min_model_size": 15,
            "ba_global_iter": 50,
            "ba_local_iter": 25,
            "use_gpu": True
        },

        "openmvg": {
            "feature_type": "SIFT",
            "matching_strategy": "ANNL2",
            "camera_model": "PINHOLE",
            "num_threads": -1,
            "sensor_database": None,
            "geometric_model": "e",
            "guided_matching": True
        }
    },

    "dense": {
        "colmap": {},
        "openmvs": {},
        "nerfstudio": {
            "method": "nerfacto",
            "fallback_order": ["nerfacto", "instant-ngp"],
            "allow_cuda_extensions": False,

            "iterations": 30000,
            "device": "cuda",
            "precision": "fp32",

            "export": {
                "type": "pointcloud",   # 🔥 explicit
                "format": "ply"
            }
        }
    },

    "mesh": {
        "colmap": {},
        "openmvs": {},
        "poisson": {}
    },

    "texture": {
        "openmvs": {}
    },

    "analysis": {
        "enabled": True,
        "save_metrics": True
    },

    "_meta": {
        "retry_count": 0,
        "last_params": None
    }
}


# =====================================================
# LOAD CONFIG
# =====================================================

def load_config(user_config=None):
    config = deepcopy(DEFAULT_CONFIG)

    if user_config:
        _deep_update(config, user_config)

    _resolve_camera_model(config)
    _validate_backends(config)
    _resolve_backend_compatibility(config)
    _validate_nerfstudio(config)

    return config


# =====================================================
# CAMERA MODEL
# =====================================================

def _resolve_camera_model(config):
    backend = config["pipeline"]["backends"]["sparse"]

    model = "PINHOLE" if backend == "openmvg" else "SIMPLE_RADIAL"

    override = config["pipeline"].get("camera_model")

    if override == "pinhole":
        model = "PINHOLE"
    elif override == "opencv":
        model = "OPENCV"

    config["pipeline"]["camera_model"] = model


# =====================================================
# BACKEND VALIDATION
# =====================================================

def _validate_backends(config):
    valid = {
        "sparse": ["colmap", "openmvg"],
        "dense": ["colmap", "openmvs", "nerfstudio"],
        "mesh": ["colmap", "openmvs", "poisson", "auto"],
    }

    for k, allowed in valid.items():
        v = config["pipeline"]["backends"].get(k)
        if v not in allowed:
            raise ValueError(f"[CONFIG] Invalid {k}: {v}")


# =====================================================
# 🔥 BACKEND COMPATIBILITY RESOLUTION (CRITICAL FIX)
# =====================================================

def _resolve_backend_compatibility(config):
    dense = config["pipeline"]["backends"]["dense"]
    mesh = config["pipeline"]["backends"]["mesh"]

    # AUTO RESOLVE
    if mesh == "auto":

        if dense == "colmap":
            config["pipeline"]["backends"]["mesh"] = "colmap"

        elif dense == "openmvs":
            config["pipeline"]["backends"]["mesh"] = "openmvs"

        elif dense == "nerfstudio":
            # depends on export type
            export_type = config["dense"]["nerfstudio"]["export"]["type"]

            if export_type == "mesh":
                config["pipeline"]["backends"]["mesh"] = "none"
            else:
                config["pipeline"]["backends"]["mesh"] = "colmap"

    # HARD VALIDATION
    dense = config["pipeline"]["backends"]["dense"]
    mesh = config["pipeline"]["backends"]["mesh"]

    if dense == "openmvs" and mesh == "colmap":
        raise RuntimeError(
            "[CONFIG ERROR] openmvs dense cannot be used with colmap mesh"
        )

    if dense == "colmap" and mesh == "openmvs":
        raise RuntimeError(
            "[CONFIG ERROR] colmap dense cannot be used with openmvs mesh"
        )


# =====================================================
# 🔥 NERF VALIDATION
# =====================================================

def _validate_nerfstudio(config):
    if config["pipeline"]["backends"]["dense"] != "nerfstudio":
        return

    ns = config["dense"]["nerfstudio"]

    if ns["device"] not in ["cuda", "cpu"]:
        raise ValueError("Invalid Nerfstudio device")

    if ns["precision"] not in ["fp32", "fp16"]:
        raise ValueError("Invalid precision")


# =====================================================
# UTIL
# =====================================================

def _deep_update(base, updates):
    for k, v in updates.items():
        if isinstance(v, dict) and k in base:
            _deep_update(base[k], v)
        else:
            base[k] = v