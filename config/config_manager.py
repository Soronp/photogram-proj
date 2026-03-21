from copy import deepcopy

DEFAULT_CONFIG = {

    # =====================================================
    # GLOBAL PIPELINE CONTROL
    # =====================================================
    "pipeline": {
        "name": "adaptive_multibackend_sfm",

        # 🔥 MASTER SWITCHES
        "mode": "mesh",  # mesh | gaussian

        "sparse_backend": "colmap",   # colmap | glomap | openmvg
        "dense_backend": "colmap",    # colmap | openmvs | none
        "mesh_backend": "colmap",     # colmap | openmvs | hybrid | none
        "texture_backend": "colmap",  # colmap | openmvs | none
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
    # FEATURE EXTRACTION (SHARED)
    # =====================================================
    "sift": {
        "max_num_features": 16000,
        "max_image_size": 3200,
        "num_threads": -1,
        "use_gpu": True,
        "peak_threshold": 0.004
    },

    # =====================================================
    # MATCHING (SHARED)
    # =====================================================
    "matching": {
        "type": "exhaustive",
        "use_gpu": True
    },

    # =====================================================
    # SPARSE BACKENDS
    # =====================================================
    "sparse": {

        # 🔥 GLOBAL CONTROL
        "backend": "colmap",
        "fallback_to_colmap": True,

        # -----------------------------
        # COLMAP PARAMETERS
        # -----------------------------
        "colmap": {
            "init_min_inliers": 40,
            "abs_min_inliers": 30,
            "min_model_size": 15,
            "ba_global_iter": 50,
            "ba_local_iter": 25,
            "use_gpu": True
        },

        # -----------------------------
        # GLOMAP PARAMETERS
        # -----------------------------
        "glomap": {
            "num_threads": -1,
            "use_rotation_averaging": True,
            "robust_loss": "Cauchy",  # placeholder for future CLI flags
        },

        # -----------------------------
        # OPENMVG (FUTURE)
        # -----------------------------
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

        # -----------------------------
        # COLMAP DENSE
        # -----------------------------
        "colmap": {
            "window_radius": 7,
            "num_samples": 25,
            "num_iterations": 7,
            "use_gpu": True
        },

        # -----------------------------
        # OPENMVS DENSE
        # -----------------------------
        "openmvs": {
            "resolution_level": 1,
            "number_views": 6,
            "use_gpu": True
        }
    },

    # =====================================================
    # FUSION (COLMAP)
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
    # LIMITS (ADAPTIVE SYSTEM)
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


def load_config(user_config=None):
    config = deepcopy(DEFAULT_CONFIG)
    if user_config:
        _deep_update(config, user_config)
    return config


def inject_analysis(config, stats):
    config["analysis_results"] = stats
    return config


def _deep_update(base, updates):
    for k, v in updates.items():
        if isinstance(v, dict) and k in base:
            _deep_update(base[k], v)
        else:
            base[k] = v