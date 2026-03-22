def run(paths, config, logger, tool_runner):
    stage = "openmvs_densify"
    logger.info(f"---- {stage.upper()} ----")

    mvs_dir = paths.run_root / "openmvs"
    scene = mvs_dir / "scene.mvs"
    dense = mvs_dir / "scene_dense.mvs"

    if not scene.exists():
        raise RuntimeError(f"{stage}: scene.mvs missing")

    resolution = config.get("dense", {}).get("openmvs", {}).get("resolution_level", 1)

    cmd = [
        "DensifyPointCloud",
        str(scene),
        "--resolution-level", str(resolution),
    ]

    tool_runner.run(cmd, stage=stage)

    if not dense.exists():
        raise RuntimeError(f"{stage}: densify failed")

    logger.info(f"{stage}: SUCCESS")