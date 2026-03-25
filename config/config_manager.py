from copy import deepcopy


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

        # SINGLE SOURCE OF TRUTH
        "backends": {
            "sparse": "colmap",   # colmap | glomap | openmvg
            "dense": "colmap",    # colmap | openmvs
            "mesh": "colmap",     # colmap | openmvs | hybrid
            "texture": "colmap"   # colmap | openmvs
        },

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
    # FEATURE EXTRACTION (COLMAP ONLY)
    # =====================================================
    "sift": {
        "max_num_features": 16000,
        "max_image_size": 3200,
        "num_threads": -1,
        "use_gpu": True,
        "peak_threshold": 0.004
    },

    # =====================================================
    # MATCHING (COLMAP ONLY)
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

        # 🔥 FULL OPENMVG CONFIG
        "openmvg": {
            "feature_type": "SIFT",
            "matching_strategy": "FASTCASCADEHASHINGL2",

            # NEW (IMPORTANT)
            "camera_model": "PINHOLE",   # PINHOLE recommended
            "num_threads": -1,

            # optional future tuning
            "geometric_model": "f",      # f, e, h
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

    return config


# =====================================================
# CAMERA MODEL LOGIC (UPDATED)
# =====================================================

def _resolve_camera_model(config):
    pipeline = config["pipeline"]
    backends = pipeline["backends"]

    sparse_backend = backends["sparse"]
    dense_backend = backends["dense"]
    mesh_backend = backends["mesh"]

    # Default: COLMAP best accuracy
    camera_model = "OPENCV"

    # OpenMVS compatibility → MUST use PINHOLE
    if dense_backend == "openmvs" or mesh_backend in ["openmvs", "hybrid"]:
        camera_model = "PINHOLE"

    # OpenMVG prefers PINHOLE (safer default)
    if sparse_backend == "openmvg":
        camera_model = "PINHOLE"

    # Explicit override
    user_choice = pipeline.get("camera_model")
    if user_choice == "pinhole":
        camera_model = "PINHOLE"
    elif user_choice == "opencv":
        camera_model = "OPENCV"

    config["pipeline"]["camera_model"] = camera_model


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