from copy import deepcopy


DEFAULT_CONFIG = {
    "pipeline": {
        "name": "baseline"
    },

    "paths": {
        "project_root": None,   # will be filled dynamically
    },

    "ingestion": {
        "copy_mode": "copy"   # "copy" or "symlink"
    },

    "downsampling": {
        "enabled": True,
        "target_max_dim": 2000
    },

    "sift": {
        "max_num_features": 8000,
        "max_image_size": 2000,
        "num_threads": -1,
        "use_gpu": False
    },

    "matching": {
        "type": "exhaustive",
        "use_gpu": False
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