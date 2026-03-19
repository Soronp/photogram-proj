def run(config, stats, logger):
    stage = "parameter_optimizer"
    logger.info(f"---- {stage.upper()} ----")

    cfg = config.copy()

    # -----------------------------
    # Extract key stats
    # -----------------------------
    num_images = stats.get("num_images", 0)
    entropy = stats.get("avg_entropy", 0)
    feature_density = stats.get("feature_density", 0)
    connectivity = stats.get("connectivity", 0)
    avg_degree = stats.get("avg_degree", 0)

    logger.info(f"{stage}: input stats -> {stats}")

    # =====================================================
    # 🔷 SIFT OPTIMIZATION
    # =====================================================
    sift = cfg.setdefault("sift", {})

    # Low texture → increase features
    if feature_density < 0.001:
        sift["max_num_features"] = 12000
        sift["peak_threshold"] = 0.004
        logger.info(f"{stage}: low feature density → boosting SIFT")

    # High texture → reduce noise
    elif feature_density > 0.005:
        sift["max_num_features"] = 6000
        sift["peak_threshold"] = 0.01
        logger.info(f"{stage}: high feature density → tightening SIFT")

    # =====================================================
    # 🔷 MATCHING STRATEGY
    # =====================================================
    matching = cfg.setdefault("matching", {})

    if connectivity < 0.15:
        matching["type"] = "exhaustive"
        logger.info(f"{stage}: poor connectivity → exhaustive matching")

    elif connectivity < 0.4:
        matching["type"] = "sequential"
        logger.info(f"{stage}: medium connectivity → sequential matching")

    else:
        matching["type"] = "vocab_tree"
        logger.info(f"{stage}: strong connectivity → vocab tree")

    # =====================================================
    # 🔷 DENSE (PATCHMATCH)
    # =====================================================
    dense = cfg.setdefault("dense", {})

    # Weak geometry → more aggressive search
    if avg_degree < 3:
        dense["window_radius"] = 9
        dense["num_samples"] = 25
        dense["num_iterations"] = 7
        logger.info(f"{stage}: weak graph → stronger PatchMatch")

    else:
        dense["window_radius"] = 5
        dense["num_samples"] = 15
        dense["num_iterations"] = 5
        logger.info(f"{stage}: stable graph → standard PatchMatch")

    # Low connectivity → relax filtering
    if connectivity < 0.2:
        dense["filter_min_num_consistent"] = 2
    else:
        dense["filter_min_num_consistent"] = 3

    # =====================================================
    # 🔷 STEREO FUSION
    # =====================================================
    fusion = cfg.setdefault("fusion", {})

    if connectivity < 0.2:
        fusion["min_num_pixels"] = 2
        logger.info(f"{stage}: sparse matches → relaxed fusion")

    else:
        fusion["min_num_pixels"] = 5
        logger.info(f"{stage}: stable matches → stricter fusion")

    # =====================================================
    # 🔷 IMAGE SCALE CONTROL
    # =====================================================
    downsampling = cfg.setdefault("downsampling", {})

    if num_images > 100:
        downsampling["target_max_dim"] = 1600
        logger.info(f"{stage}: large dataset → stronger downsampling")

    elif num_images < 40:
        downsampling["target_max_dim"] = 2400
        logger.info(f"{stage}: small dataset → preserve resolution")

    # =====================================================
    # 🔷 FINAL LOG
    # =====================================================
    logger.info(f"{stage}: optimized config ready")

    return cfg