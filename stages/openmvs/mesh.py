def run(paths, config, logger, tool_runner):
    stage = "openmvs_mesh"
    logger.info(f"---- {stage.upper()} ----")

    mvs_dir = paths.run_root / "openmvs"
    dense = mvs_dir / "scene_dense.mvs"
    mesh = mvs_dir / "scene_dense_mesh.mvs"

    if not dense.exists():
        raise RuntimeError(f"{stage}: dense scene missing")

    cmd = [
        "ReconstructMesh",
        str(dense),
    ]

    tool_runner.run(cmd, stage=stage)

    if not mesh.exists():
        raise RuntimeError(f"{stage}: mesh failed")

    logger.info(f"{stage}: SUCCESS")