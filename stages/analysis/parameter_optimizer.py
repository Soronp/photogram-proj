from copy import deepcopy


def _safe_get(d, key, default):
    """Safe dictionary access."""
    return d.get(key, default)


def _compute_total_pixels(ds):
    """Compute total pixels in the dataset."""
    if "total_pixels" in ds:
        return ds["total_pixels"]

    w = ds.get("avg_width", 0)
    h = ds.get("avg_height", 0)
    n = ds.get("num_images", 0)
    return int(w * h * n)


def _classify_dataset(num_images, total_pixels):
    """Classify dataset size based on number of images and total pixels."""
    if num_images <= 40 and total_pixels < 5e8:
        return "small"
    elif num_images <= 150 and total_pixels < 1.5e9:
        return "medium"
    else:
        return "large"


def run(config, stats, logger):
    """
    Compute optimal pipeline parameters based on dataset statistics.
    Returns a dictionary of updates to merge into the main config.
    """
    stage = "parameter_optimizer"
    logger.info(f"---- {stage.upper()} ----")

    cfg = deepcopy(config)
    ds = stats.get("dataset", {})
    ms = stats.get("matches", {})

    num_images = _safe_get(ds, "num_images", 0)
    total_pixels = _compute_total_pixels(ds)
    size = ds.get("size_class") or _classify_dataset(num_images, total_pixels)
    connectivity = _safe_get(ms, "connectivity", 0.0)

    logger.info(f"{stage}: num_images = {num_images}")
    logger.info(f"{stage}: total_pixels = {total_pixels}")
    logger.info(f"{stage}: dataset size = {size}")
    logger.info(f"{stage}: connectivity = {connectivity}")

    updates = {}

    # -----------------------------
    # Backend selection (automatic)
    # -----------------------------
    backend = "glomap" if (num_images > 120 and connectivity > 0.4) else "colmap"
    updates["sparse"] = {"backend": backend}
    logger.info(f"{stage}: selected sparse backend = {backend}")

    # -----------------------------
    # Auto downsampling
    # -----------------------------
    downsample = False
    target_dim = 2400
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
    logger.info(f"{stage}: downsampling = {downsample} (target_dim={target_dim})")

    # -----------------------------
    # High-first profile
    # -----------------------------
    if size == "small":
        updates.update({
            "sift": {"max_num_features": 20000, "peak_threshold": 0.004},
            "matching": {"type": "exhaustive"},
            "dense": {"window_radius": 9, "num_samples": 30, "num_iterations": 7, "filter_min_num_consistent": 2, "use_gpu": True},
            "fusion": {"min_num_pixels": 2},
            "mesh": {"poisson_depth": 12}
        })
    elif size == "medium":
        updates.update({
            "sift": {"max_num_features": 16000, "peak_threshold": 0.0045},
            "matching": {"type": "exhaustive"},
            "dense": {"window_radius": 8, "num_samples": 25, "num_iterations": 6, "filter_min_num_consistent": 2, "use_gpu": True},
            "fusion": {"min_num_pixels": 2},
            "mesh": {"poisson_depth": 11}
        })
    else:  # large
        updates.update({
            "sift": {"max_num_features": 10000, "peak_threshold": 0.005},
            "matching": {"type": "sequential"},
            "dense": {"window_radius": 6, "num_samples": 18, "num_iterations": 5, "filter_min_num_consistent": 3, "use_gpu": True},
            "fusion": {"min_num_pixels": 3},
            "mesh": {"poisson_depth": 10}
        })

    logger.info(f"{stage}: HIGH-FIRST profile applied ({size})")
    return updates