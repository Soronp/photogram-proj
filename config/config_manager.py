from copy import deepcopy

# =====================================================
# DEFAULT CONFIG
# =====================================================

DEFAULT_CONFIG = {

    # =====================================================
    # GLOBAL PIPELINE CONTROL
    # =====================================================
    "pipeline": {
        "name": "adaptive_multibackend_sfm",

        # master mode
        "mode": "mesh",

        "sparse_backend": "colmap",
        "dense_backend": "colmap",
        "mesh_backend": "colmap",
        "texture_backend": "colmap",

        # camera model policy
        "camera_model": "auto"   # auto | pinhole | opencv
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
    # FEATURE EXTRACTION
    # =====================================================
    "sift": {
        "max_num_features": 16000,
        "max_image_size": 3200,
        "num_threads": -1,
        "use_gpu": True,
        "peak_threshold": 0.004
    },

    # =====================================================
    # MATCHING
    # =====================================================
    "matching": {
        "type": "exhaustive",
        "use_gpu": True
    },

    # =====================================================
    # SPARSE BACKENDS
    # =====================================================
    "sparse": {
        "backend": "colmap",
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
            "matching_strategy": "FASTCASCADEHASHINGL2",
        }
    },

    # =====================================================
    # DENSE BACKENDS
    # =====================================================
    "dense": {
        "backend": "colmap",

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
        "backend": "colmap",

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
        "backend": "colmap",

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
    # LIMITS
    # =====================================================
    "limits": {
        "sift_max_features": [8000, 20000],
        "window_radius": [3, 9],
        "num_samples": [10, 30],
        "min_num_pixels": [2, 10]
    },

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

    # =====================================================
    # 🔥 PIPELINE-AWARE CAMERA MODEL LOGIC
    # =====================================================

    pipeline = config["pipeline"]

    # default: best COLMAP accuracy
    camera_model = "OPENCV"

    # Pipeline C → OpenMVS compatibility mode
    if pipeline["mesh_backend"] in ["openmvs", "hybrid"] or pipeline["dense_backend"] == "openmvs":
        camera_model = "PINHOLE"

    # explicit override
    if pipeline.get("camera_model") == "pinhole":
        camera_model = "PINHOLE"
    elif pipeline.get("camera_model") == "opencv":
        camera_model = "OPENCV"

    config["pipeline"]["camera_model"] = camera_model

    return config


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