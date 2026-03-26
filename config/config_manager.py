from copy import deepcopy
from pathlib import Path


# =====================================================
# DEFAULT CONFIG
# =====================================================

DEFAULT_CONFIG = {

    # =====================================================
    # PIPELINE CONTROL
    # =====================================================
    "pipeline": {
        "name": "adaptive_multibackend_sfm",
        "mode": "mesh",

        "backends": {
            "sparse": "colmap",   # colmap | glomap | openmvg
            "dense": "colmap",    # colmap | openmvs
            "mesh": "colmap",     # colmap | openmvs | hybrid
            "texture": "colmap"
        },

        # auto-resolved later
        "camera_model": "auto"
    },

    # =====================================================
    # PATHS
    # =====================================================
    "paths": {
        "project_root": None,
    },

    # =====================================================
    # INGESTION
    # =====================================================
    "ingestion": {
        "copy_mode": "copy"
    },

    # =====================================================
    # DOWNSAMPLING
    # =====================================================
    "downsampling": {
        "enabled": True,
        "target_max_dim": 2400
    },

    # =====================================================
    # COLMAP FEATURES
    # =====================================================
    "sift": {
        "max_num_features": 16000,
        "max_image_size": 3200,
        "num_threads": -1,
        "use_gpu": True,
        "peak_threshold": 0.004
    },

    # =====================================================
    # COLMAP MATCHING
    # =====================================================
    "matching": {
        "type": "exhaustive",
        "use_gpu": True
    },

    # =====================================================
    # SPARSE BACKENDS
    # =====================================================
    "sparse": {

        "fallback_to_colmap": True,

        # -------------------------
        # COLMAP
        # -------------------------
        "colmap": {
            "init_min_inliers": 40,
            "abs_min_inliers": 30,
            "min_model_size": 15,
            "ba_global_iter": 50,
            "ba_local_iter": 25,
            "use_gpu": True
        },

        # -------------------------
        # GLOMAP
        # -------------------------
        "glomap": {
            "num_threads": -1,
            "use_rotation_averaging": True,
            "robust_loss": "Cauchy",
        },

        # -------------------------
        # OPENMVG (🔥 FIXED)
        # -------------------------
        "openmvg": {
            "feature_type": "SIFT",
            "matching_strategy": "ANNL2",   # stronger default

            # CAMERA SETTINGS
            "camera_model": "PINHOLE",     # safest for OpenMVG
            "num_threads": -1,

            # 🔥 CRITICAL FIX (YOUR PATH)
            "sensor_database": "D:/CSE499_MK-2/OpenMVG/sensor_width_database/sensor_width_camera_database.txt",

            # MATCHING / GEOMETRY
            "geometric_model": "e",        # e = essential matrix (better default)
            "guided_matching": True,

            # FALLBACK IF DB FAILS
            "fallback_focal_multiplier": 1.2
        }
    },

    # =====================================================
    # DENSE BACKENDS
    # =====================================================
    "dense": {

        "colmap": {
            "window_radius": 7,
            "num_samples": 25,
            "num_iterations": 7,
            "use_gpu": True
        },

        "openmvs": {
            "resolution_level": 1,
            "number_views": 6,
            "use_gpu": True
        }
    },

    # =====================================================
    # FUSION
    # =====================================================
    "fusion": {
        "min_num_pixels": 2
    },

    # =====================================================
    # MESH BACKENDS
    # =====================================================
    "mesh": {

        "colmap": {},

        "openmvs": {
            "min_face_area": 16
        },

        "hybrid": {
            "poisson_depth": 10,
            "bpa_radius": 0.02
        }
    },

    # =====================================================
    # TEXTURE BACKENDS
    # =====================================================
    "texture": {

        "openmvs": {
            "resolution": 4096
        }
    },

    # =====================================================
    # GAUSSIAN SPLATTING
    # =====================================================
    "gaussian": {
        "enabled": False,
        "iterations": 30000,
        "resolution": 1.0
    },

    # =====================================================
    # ANALYSIS
    # =====================================================
    "analysis": {
        "enabled": True,
        "save_metrics": True
    },

    "analysis_results": {},

    # =====================================================
    # META
    # =====================================================
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
    _validate_openmvg(config)

    return config


# =====================================================
# CAMERA MODEL RESOLUTION
# =====================================================

def _resolve_camera_model(config):
    pipeline = config["pipeline"]
    backends = pipeline["backends"]

    sparse_backend = backends["sparse"]
    dense_backend = backends["dense"]
    mesh_backend = backends["mesh"]

    camera_model = "OPENCV"

    if dense_backend == "openmvs" or mesh_backend in ["openmvs", "hybrid"]:
        camera_model = "PINHOLE"

    if sparse_backend == "openmvg":
        camera_model = "PINHOLE"

    user_choice = pipeline.get("camera_model")

    if user_choice == "pinhole":
        camera_model = "PINHOLE"
    elif user_choice == "opencv":
        camera_model = "OPENCV"

    config["pipeline"]["camera_model"] = camera_model


# =====================================================
# 🔥 OPENMVG VALIDATION (NEW)
# =====================================================

def _validate_openmvg(config):
    sparse_backend = config["pipeline"]["backends"]["sparse"]

    if sparse_backend != "openmvg":
        return

    openmvg_cfg = config["sparse"]["openmvg"]
    sensor_db = openmvg_cfg.get("sensor_database")

    if sensor_db:
        db_path = Path(sensor_db)
        if not db_path.exists():
            raise FileNotFoundError(
                f"[OpenMVG] Sensor database not found: {sensor_db}"
            )


# =====================================================
# ANALYSIS INJECTION
# =====================================================

def inject_analysis(config, stats):
    config["analysis_results"] = stats
    return config


# =====================================================
# DEEP UPDATE
# =====================================================

def _deep_update(base, updates):
    for k, v in updates.items():
        if isinstance(v, dict) and k in base:
            _deep_update(base[k], v)
        else:
            base[k] = v