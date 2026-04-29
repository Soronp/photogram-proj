from copy import deepcopy

# =====================================================
# DEFAULT CONFIG (CLEAN PIPELINE ENGINE)
# =====================================================

DEFAULT_CONFIG = {

    "pipeline": {
        "name": "adaptive_multibackend_sfm",
        "mode": "full",

        "backends": {
            "sparse": "colmap",     # colmap | openmvg
            "dense": "colmap",      # colmap | openmvs | nerfstudio
            "mesh": "colmap",       # colmap | openmvs | poisson
            "texture": "colmap"     # colmap | openmvs
        },

        "camera_model": "auto"
    },

    "paths": {"project_root": None},

    "ingestion": {"copy_mode": "copy"},

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

    # =====================================================
    # BACKEND SPECIFICATIONS
    # =====================================================

    "sparse": {
        "colmap": {},
        "openmvg": {
            "feature_type": "SIFT",
            "matching_strategy": "ANNL2",
            "camera_model": "PINHOLE",
            "guided_matching": True
        }
    },

    "dense": {
        "colmap": {},
        "openmvs": {},
        "nerfstudio": {
            "method": "nerfacto",
            "iterations": 30000,
            "device": "cuda",
            "precision": "fp32",
            "export": {
                "type": "pointcloud",
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
        "colmap": {},
        "openmvs": {}
    },

    "analysis": {
        "enabled": True,
        "save_metrics": True
    }
}


# =====================================================
# LOAD CONFIG
# =====================================================

def load_config(user_config=None):
    config = deepcopy(DEFAULT_CONFIG)

    if user_config:
        _deep_update(config, user_config)

    _resolve_pipeline_rules(config)
    _validate_backends(config)
    _resolve_camera_model(config)

    return config


# =====================================================
# PIPELINE RULE ENGINE (CORE FIX)
# =====================================================

def _resolve_pipeline_rules(config):
    """
    This is the ONLY place pipeline behavior is defined.
    No None hacks. No ambiguous optional logic.
    """

    backends = config["pipeline"]["backends"]
    sparse = backends["sparse"]

    # -------------------------------------------------
    # PIPELINE D = OpenMVG + OpenMVS FULL STACK
    # -------------------------------------------------
    if sparse == "openmvg":
        backends["dense"] = "openmvs"
        backends["mesh"] = "openmvs"
        backends["texture"] = "openmvs"
        return

    # -------------------------------------------------
    # COLMAP FULL PIPELINE
    # -------------------------------------------------
    if sparse == "colmap":
        backends["dense"] = "colmap"
        backends["mesh"] = "colmap"
        backends["texture"] = "colmap"


# =====================================================
# VALIDATION (STRICT BUT CONSISTENT)
# =====================================================

def _validate_backends(config):
    valid = {
        "sparse": ["colmap", "openmvg"],
        "dense": ["colmap", "openmvs", "nerfstudio"],
        "mesh": ["colmap", "openmvs", "poisson"],
        "texture": ["colmap", "openmvs"]
    }

    for k, allowed in valid.items():
        v = config["pipeline"]["backends"][k]
        if v not in allowed:
            raise ValueError(f"[CONFIG] Invalid {k}: {v}")


# =====================================================
# CAMERA MODEL
# =====================================================

def _resolve_camera_model(config):
    sparse = config["pipeline"]["backends"]["sparse"]

    config["pipeline"]["camera_model"] = (
        "PINHOLE" if sparse == "openmvg" else "SIMPLE_RADIAL"
    )


# =====================================================
# UTIL
# =====================================================

def _deep_update(base, updates):
    for k, v in updates.items():
        if isinstance(v, dict) and k in base:
            _deep_update(base[k], v)
        else:
            base[k] = v