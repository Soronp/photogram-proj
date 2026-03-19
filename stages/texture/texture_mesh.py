from pathlib import Path


def run(paths, config, logger, tool_runner):
    stage = "texture_mesh"
    logger.info(f"---- {stage.upper()} ----")

    sparse_root = paths.sparse   # 🔥 FIXED
    image_dir = paths.images_downsampled if config.get("downsampling", {}).get("enabled", True) else paths.images

    if not sparse_root.exists():
        raise RuntimeError(f"{stage}: sparse folder not found")

    # -----------------------------
    # OpenMVS workspace
    # -----------------------------
    mvs_dir = paths.run_root / "openmvs"
    mvs_dir.mkdir(parents=True, exist_ok=True)

    scene_mvs = mvs_dir / "scene.mvs"

    # -----------------------------
    # Step 1: InterfaceCOLMAP
    # -----------------------------
    if not scene_mvs.exists():
        logger.info(f"{stage}: converting COLMAP -> OpenMVS")

        cmd = [
            "InterfaceCOLMAP",
            "-i", str(sparse_root),   # 🔥 FIXED
            "-o", str(scene_mvs),
            "-w", str(mvs_dir),
        ]

        tool_runner.run(cmd, stage=stage + "_interface")

    # -----------------------------
    # Step 2: Densify
    # -----------------------------
    dense_mvs = mvs_dir / "scene_dense.mvs"

    if not dense_mvs.exists():
        logger.info(f"{stage}: densifying point cloud")

        cmd = [
            "DensifyPointCloud",
            str(scene_mvs),
            "--resolution-level", "1",  # 🔥 better quality
        ]

        tool_runner.run(cmd, stage=stage + "_densify")

    # -----------------------------
    # Step 3: Mesh
    # -----------------------------
    mesh_mvs = mvs_dir / "scene_dense_mesh.mvs"

    if not mesh_mvs.exists():
        logger.info(f"{stage}: reconstructing mesh")

        cmd = [
            "ReconstructMesh",
            str(dense_mvs),
        ]

        tool_runner.run(cmd, stage=stage + "_mesh")

    # -----------------------------
    # Step 4: Texture
    # -----------------------------
    textured_obj = mvs_dir / "scene_dense_mesh_texture.obj"

    if not textured_obj.exists():
        logger.info(f"{stage}: texturing mesh")

        cmd = [
            "TextureMesh",
            str(mesh_mvs),
        ]

        tool_runner.run(cmd, stage=stage + "_texture")

    logger.info(f"{stage}: textured mesh ready at {textured_obj}")