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
        "mode_variant": "default",

        "backends": {
            "sparse": "colmap",
            "dense": "colmap",
            "mesh": "colmap",
            "texture": "colmap"
        },

        # auto-resolved later
        "camera_model": "auto"
    },

    # =====================================================
    # PATHS
    # =====================================================
    "paths": {
        "project_root": None
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

        "colmap": {
            "init_min_inliers": 40,
            "abs_min_inliers": 30,
            "min_model_size": 15,
            "ba_global_iter": 50,
            "ba_local_iter": 25,
            "use_gpu": True
        },

        "glomap": {
            "num_threads": -1,
            "use_rotation_averaging": True,
            "robust_loss": "Cauchy",
        },

        "openmvg": {
            "feature_type": "SIFT",
            "matching_strategy": "ANNL2",

            "camera_model": "PINHOLE",
            "num_threads": -1,

            "sensor_database": "D:/CSE499_MK-2/OpenMVG/sensor_width_database/sensor_width_camera_database.txt",

            "geometric_model": "e",
            "guided_matching": True,

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
            "use_gpu": True,

            "pipeline_D": {
                "resolution_level": 0,
                "number_views": 8,
                "number_views_fuse": 5,
                "fusion_filter": 0,
                "filter_point_cloud": 0,
                "estimate_colors": 2,
                "estimate_normals": 2,
                "min_resolution": 640,
                "max_resolution": 6000,
                "sub_resolution_levels": 2
            }
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
# CAMERA MODEL RESOLUTION (FIXED)
# =====================================================
def _resolve_camera_model(config):
    pipeline = config["pipeline"]
    backends = pipeline["backends"]

    sparse_backend = backends["sparse"]

    # -------------------------------------------------
    # 🔥 CORRECT LOGIC: BASED ON SPARSE ONLY
    # -------------------------------------------------
    if sparse_backend in ["colmap", "glomap"]:
        camera_model = "SIMPLE_RADIAL"

    elif sparse_backend == "openmvg":
        camera_model = "PINHOLE"

    else:
        camera_model = "SIMPLE_RADIAL"

    # -------------------------------------------------
    # 🔥 USER OVERRIDE (RESPECTED)
    # -------------------------------------------------
    user_choice = pipeline.get("camera_model")

    if user_choice == "pinhole":
        camera_model = "PINHOLE"

    elif user_choice == "opencv":
        camera_model = "OPENCV"

    elif user_choice == "simple_radial":
        camera_model = "SIMPLE_RADIAL"

    config["pipeline"]["camera_model"] = camera_model


# =====================================================
# OPENMVG VALIDATION
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