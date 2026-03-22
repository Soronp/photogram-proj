from pathlib import Path


def run(paths, config, logger, tool_runner):
    stage = "texture_mesh"
    logger.info(f"---- {stage.upper()} ----")

    dense_dir = paths.dense

    fused_path = dense_dir / "fused.ply"
    mesh_path = dense_dir / "mesh.ply"
    textured_path = dense_dir / "textured.obj"

    if not fused_path.exists():
        raise RuntimeError(f"{stage}: fused point cloud missing")

    # =====================================================
    # 🔥 STEP 1: MESH (Poisson or Delaunay)
    # =====================================================
    meshing_method = config.get("mesh", {}).get("method", "poisson")

    if mesh_path.exists():
        mesh_path.unlink()

    if meshing_method == "poisson":
        logger.info(f"{stage}: using Poisson meshing")

        tool_runner.run([
            "colmap",
            "poisson_mesher",
            "--input_path", str(fused_path),
            "--output_path", str(mesh_path),
            "--PoissonMeshing.trim", "10",
        ], stage=stage + "_poisson")

    else:
        logger.info(f"{stage}: using Delaunay meshing")

        tool_runner.run([
            "colmap",
            "delaunay_mesher",
            "--input_path", str(dense_dir),
            "--output_path", str(mesh_path),
        ], stage=stage + "_delaunay")

    if not mesh_path.exists():
        raise RuntimeError(f"{stage}: meshing failed")

    # =====================================================
    # 🔥 STEP 2: TEXTURE
    # =====================================================
    if textured_path.exists():
        textured_path.unlink()

    logger.info(f"{stage}: texturing mesh")

    tool_runner.run([
        "colmap",
        "texture_mesher",
        "--input_path", str(dense_dir),
        "--input_mesh_path", str(mesh_path),
        "--output_path", str(textured_path),
        "--TextureMesher.num_threads", "-1",
        "--TextureMesher.export_obj", "1",
    ], stage=stage + "_texture")

    if not textured_path.exists():
        raise RuntimeError(f"{stage}: texturing failed")

    logger.info(f"{stage}: SUCCESS → {textured_path}")