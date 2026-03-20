from copy import deepcopy


def run(config, stats, logger):
    stage = "parameter_optimizer"
    logger.info(f"---- {stage.upper()} ----")

    cfg = deepcopy(config)

    ds = stats.get("dataset", {})
    fs = stats.get("features", {})
    ms = stats.get("matches", {})

    num_images = ds.get("num_images", 0)
    total_pixels = ds.get("total_pixels", 0)
    size = ds.get("size_class", "medium")

    connectivity = ms.get("connectivity", 0)

    logger.info(f"{stage}: dataset size = {size}")
    logger.info(f"{stage}: total_pixels = {total_pixels}")

    updates = {}

    # =====================================================
    # 🔥 BACKEND SELECTION (STABLE RULE)
    # =====================================================
    if num_images > 120 and connectivity > 0.4:
        backend = "glomap"
        logger.info(f"{stage}: using GLOMAP")
    else:
        backend = "colmap"
        logger.info(f"{stage}: using COLMAP")

    updates["sparse"] = {"backend": backend}

    # =====================================================
    # 🔥 DOWNSAMPLING DECISION (HARDWARE-BASED)
    # =====================================================
    downsample = False

    if size == "large" or total_pixels > 1.5e9:
        downsample = True

    updates["downsampling"] = {
        "enabled": downsample,
        "target_max_dim": 1600 if downsample else 2400
    }

    # =====================================================
    # 🔷 PROFILE SELECTION
    # =====================================================

    # -----------------------------
    # SMALL DATASET → MAX QUALITY
    # -----------------------------
    if size == "small":

        updates["sift"] = {
            "max_num_features": 18000,
            "peak_threshold": 0.004
        }

        updates["matching"] = {
            "type": "exhaustive"
        }

        updates["dense"] = {
            "window_radius": 9,
            "num_samples": 30,
            "num_iterations": 7,
            "filter_min_num_consistent": 2
        }

        updates["fusion"] = {
            "min_num_pixels": 2
        }

        updates["mesh"] = {
            "poisson_depth": 12
        }

    # -----------------------------
    # MEDIUM DATASET → BALANCED
    # -----------------------------
    elif size == "medium":

        updates["sift"] = {
            "max_num_features": 12000,
            "peak_threshold": 0.005
        }

        updates["matching"] = {
            "type": "sequential"
        }

        updates["dense"] = {
            "window_radius": 7,
            "num_samples": 20,
            "num_iterations": 5,
            "filter_min_num_consistent": 3
        }

        updates["fusion"] = {
            "min_num_pixels": 3
        }

        updates["mesh"] = {
            "poisson_depth": 11
        }

    # -----------------------------
    # LARGE DATASET → SAFE MODE
    # -----------------------------
    else:

        updates["sift"] = {
            "max_num_features": 8000,
            "peak_threshold": 0.006
        }

        updates["matching"] = {
            "type": "vocab_tree"
        }

        updates["dense"] = {
            "window_radius": 5,
            "num_samples": 15,
            "num_iterations": 4,
            "filter_min_num_consistent": 3
        }

        updates["fusion"] = {
            "min_num_pixels": 5
        }

        updates["mesh"] = {
            "poisson_depth": 10
        }

    logger.info(f"{stage}: profile applied ({size})")

    return updates