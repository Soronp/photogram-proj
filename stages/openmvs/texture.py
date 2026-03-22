def run(paths, config, logger, tool_runner):
    stage = "openmvs_texture"
    logger.info(f"---- {stage.upper()} ----")

    mvs_dir = paths.run_root / "openmvs"
    mesh = mvs_dir / "scene_dense_mesh.mvs"
    output = mvs_dir / "scene_dense_mesh_texture.obj"

    if not mesh.exists():
        raise RuntimeError(f"{stage}: mesh missing")

    cmd = [
        "TextureMesh",
        str(mesh),
    ]

    tool_runner.run(cmd, stage=stage)

    if not output.exists():
        raise RuntimeError(f"{stage}: texture failed")

    logger.info(f"{stage}: SUCCESS → {output}")