from copy import deepcopy


DEFAULT_CONFIG = {
    "pipeline": {
        "name": "baseline"
    },

    "paths": {
        "project_root": None,
    },

    "ingestion": {
        "copy_mode": "copy"
    },

    "downsampling": {
        "enabled": True,
        "target_max_dim": 2000
    },

    # -----------------------------
    # SIFT / FEATURES
    # -----------------------------
    "sift": {
        "max_num_features": 8000,
        "max_image_size": 2000,
        "num_threads": -1,
        "use_gpu": False
    },

    # -----------------------------
    # MATCHING
    # -----------------------------
    "matching": {
        "type": "exhaustive",
        "use_gpu": False
    },

    # -----------------------------
    # SPARSE (SfM)
    # -----------------------------
    "sparse": {
        "backend": "colmap"  # or "glomap"
    },

    # -----------------------------
    # DENSE (COLMAP)
    # -----------------------------
    "dense": {
        "max_image_size": 2000,
        "use_gpu": False,
        "num_threads": -1,
        "window_radius": 5,
        "num_samples": 15,
        "min_num_pixels": 5
    },

    # -----------------------------
    # MESH
    # -----------------------------
    "mesh": {
        "method": "poisson",
        "poisson_depth": 9,
        "voxel_size": 0.01
    }
}


def load_config(user_config: dict = None):
    config = deepcopy(DEFAULT_CONFIG)

    if user_config:
        _deep_update(config, user_config)

    return config


def _deep_update(base, updates):
    for k, v in updates.items():
        if isinstance(v, dict) and k in base:
            _deep_update(base[k], v)
        else:
            base[k] = v