from copy import deepcopy


def _safe_get(d, key, default):
    return d[key] if key in d else default


def _compute_total_pixels(ds):
    if "total_pixels" in ds:
        return ds["total_pixels"]

    w = ds.get("avg_width", 0)
    h = ds.get("avg_height", 0)
    n = ds.get("num_images", 0)

    return int(w * h * n)


def _classify_dataset(num_images, total_pixels):
    """
    Hardware-aware classification (YOUR MACHINE)
    Ryzen 5 + 16GB RAM + RTX
    """

    if num_images <= 40 and total_pixels < 5e8:
        return "small"

    elif num_images <= 150 and total_pixels < 1.5e9:
        return "medium"

    else:
        return "large"


def run(config, stats, logger):
    stage = "parameter_optimizer"
    logger.info(f"---- {stage.upper()} ----")

    cfg = deepcopy(config)

    # =====================================================
    # 🔥 SAFE EXTRACTION
    # =====================================================
    ds = stats.get("dataset", {})
    fs = stats.get("features", {})
    ms = stats.get("matches", {})

    # 🔥 CRITICAL FIX: always safe
    num_images = _safe_get(ds, "num_images", 0)

    total_pixels = _compute_total_pixels(ds)
    size = ds.get("size_class")

    if not size:
        size = _classify_dataset(num_images, total_pixels)

    connectivity = _safe_get(ms, "connectivity", 0.0)

    logger.info(f"{stage}: num_images = {num_images}")
    logger.info(f"{stage}: total_pixels = {total_pixels}")
    logger.info(f"{stage}: dataset size = {size}")
    logger.info(f"{stage}: connectivity = {connectivity}")

    updates = {}

    # =====================================================
    # 🔥 BACKEND SELECTION (STABLE)
    # =====================================================
    if num_images > 120 and connectivity > 0.4:
        backend = "glomap"
        logger.info(f"{stage}: using GLOMAP")
    else:
        backend = "colmap"
        logger.info(f"{stage}: using COLMAP")

    updates["sparse"] = {"backend": backend}

    # =====================================================
    # 🔥 AUTO DOWNSAMPLING (YOUR HARDWARE LIMIT)
    # =====================================================
    downsample = False
    target_dim = 2400  # default HIGH quality

    if size == "large" or total_pixels > 1.5e9:
        downsample = True
        target_dim = 1600

    elif size == "medium" and total_pixels > 8e8:
        downsample = True
        target_dim = 1800

    updates["downsampling"] = {
        "enabled": downsample,
        "target_max_dim": target_dim
    }

    logger.info(f"{stage}: downsampling = {downsample} ({target_dim})")

    # =====================================================
    # 🔥 HIGH-FIRST STRATEGY (CRITICAL CHANGE)
    # =====================================================
    # Instead of "safe defaults", we PUSH hardware limits

    # -----------------------------
    # SMALL → MAX QUALITY
    # -----------------------------
    if size == "small":

        updates["sift"] = {
            "max_num_features": 20000,
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
    # MEDIUM → HIGH (NOT BALANCED)
    # -----------------------------
    elif size == "medium":

        updates["sift"] = {
            "max_num_features": 16000,
            "peak_threshold": 0.0045
        }

        updates["matching"] = {
            "type": "exhaustive"
        }

        updates["dense"] = {
            "window_radius": 8,
            "num_samples": 25,
            "num_iterations": 6,
            "filter_min_num_consistent": 2
        }

        updates["fusion"] = {
            "min_num_pixels": 2
        }

        updates["mesh"] = {
            "poisson_depth": 11
        }

    # -----------------------------
    # LARGE → CONTROLLED HIGH
    # -----------------------------
    else:

        updates["sift"] = {
            "max_num_features": 10000,
            "peak_threshold": 0.005
        }

        updates["matching"] = {
            "type": "sequential"
        }

        updates["dense"] = {
            "window_radius": 6,
            "num_samples": 18,
            "num_iterations": 5,
            "filter_min_num_consistent": 3
        }

        updates["fusion"] = {
            "min_num_pixels": 3
        }

        updates["mesh"] = {
            "poisson_depth": 10
        }

    logger.info(f"{stage}: HIGH-FIRST profile applied ({size})")

    return updates