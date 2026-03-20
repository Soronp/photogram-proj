from copy import deepcopy


DEFAULT_CONFIG = {

    "pipeline": {
        "name": "adaptive_hybrid_sfm"
    },

    "paths": {
        "project_root": None,
    },

    # -----------------------------
    # INGESTION
    # -----------------------------
    "ingestion": {
        "copy_mode": "copy"
    },

    "downsampling": {
        "enabled": True,
        "target_max_dim": 2000
    },

    # -----------------------------
    # SIFT
    # -----------------------------
    "sift": {
        "max_num_features": 8000,
        "max_image_size": 2000,
        "num_threads": -1,
        "use_gpu": False,
        "peak_threshold": 0.006
    },

    # -----------------------------
    # MATCHING
    # -----------------------------
    "matching": {
        "type": "exhaustive",
        "use_gpu": False
    },

    # -----------------------------
    # SPARSE BACKEND
    # -----------------------------
    "sparse": {
        "backend": "colmap",   # or "glomap"
        "fallback_to_colmap": True
    },

    # -----------------------------
    # DENSE
    # -----------------------------
    "dense": {
        "window_radius": 5,
        "num_samples": 15,
        "num_iterations": 5,
        "min_num_pixels": 5,
        "use_gpu": True
    },

    # -----------------------------
    # FUSION
    # -----------------------------
    "fusion": {
        "min_num_pixels": 5
    },

    # -----------------------------
    # ANALYSIS CONTROL
    # -----------------------------
    "analysis": {
        "enabled": True,
        "save_metrics": True
    },

    # 🔥 RUNTIME ANALYSIS STORAGE
    "analysis_results": {},

    # -----------------------------
    # LIMITS
    # -----------------------------
    "limits": {
        "sift_max_features": [4000, 20000],
        "window_radius": [3, 9],
        "num_samples": [10, 30],
        "min_num_pixels": [2, 10]
    },

    # 🔥 META (for retries)
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