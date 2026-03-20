from pathlib import Path
import shutil


def build_colmap_workspace(paths, logger):
    """
    Build a proper COLMAP workspace for OpenMVS:
    workspace/
        images/
        sparse/0/
    """

    workspace = paths.run_root / "mvs_workspace"

    if workspace.exists():
        shutil.rmtree(workspace)

    workspace.mkdir(parents=True)

    # -----------------------------
    # Copy images
    # -----------------------------
    src_images = paths.images_downsampled if paths.images_downsampled.exists() else paths.images
    dst_images = workspace / "images"

    shutil.copytree(src_images, dst_images)

    # -----------------------------
    # Copy sparse model
    # -----------------------------
    src_sparse = paths.sparse_model

    if not src_sparse.exists():
        raise RuntimeError("texture_mesh: sparse model (0) not found")

    dst_sparse = workspace / "sparse" / "0"
    dst_sparse.parent.mkdir(parents=True)

    shutil.copytree(src_sparse, dst_sparse)

    logger.info(f"texture_mesh: workspace created at {workspace}")

    return workspace


def run(paths, config, logger, tool_runner):
    stage = "texture_mesh"
    logger.info(f"---- {stage.upper()} ----")

    if not paths.sparse_model.exists():
        raise RuntimeError(f"{stage}: sparse model not found")

    # =====================================================
    # 🔥 BUILD CLEAN WORKSPACE (CRITICAL FIX)
    # =====================================================
    workspace = build_colmap_workspace(paths, logger)

    # =====================================================
    # OpenMVS OUTPUT DIR
    # =====================================================
    mvs_dir = paths.texture
    mvs_dir.mkdir(parents=True, exist_ok=True)

    scene_mvs = mvs_dir / "scene.mvs"
    dense_mvs = mvs_dir / "scene_dense.mvs"
    mesh_mvs = mvs_dir / "scene_dense_mesh.mvs"
    textured_obj = mvs_dir / "scene_dense_mesh_texture.obj"

    # 🔥 Always clean stale outputs (important for retries)
    for f in [scene_mvs, dense_mvs, mesh_mvs, textured_obj]:
        if f.exists():
            f.unlink()

    # =====================================================
    # STEP 1: InterfaceCOLMAP
    # =====================================================
    logger.info(f"{stage}: InterfaceCOLMAP")

    cmd = [
        "InterfaceCOLMAP",
        "-i", str(workspace),   # ✅ CORRECT
        "-o", str(scene_mvs),
        "-w", str(mvs_dir),
    ]

    tool_runner.run(cmd, stage=stage + "_interface")

    if not scene_mvs.exists():
        raise RuntimeError(f"{stage}: InterfaceCOLMAP failed")

    # =====================================================
    # STEP 2: DENSIFY
    # =====================================================
    logger.info(f"{stage}: DensifyPointCloud")

    cmd = [
        "DensifyPointCloud",
        str(scene_mvs),
        "--resolution-level", "1",
    ]

    tool_runner.run(cmd, stage=stage + "_densify")

    if not dense_mvs.exists():
        raise RuntimeError(f"{stage}: densify failed")

    # =====================================================
    # STEP 3: MESH
    # =====================================================
    logger.info(f"{stage}: ReconstructMesh")

    cmd = [
        "ReconstructMesh",
        str(dense_mvs),
    ]

    tool_runner.run(cmd, stage=stage + "_mesh")

    if not mesh_mvs.exists():
        raise RuntimeError(f"{stage}: mesh failed")

    # =====================================================
    # STEP 4: TEXTURE
    # =====================================================
    logger.info(f"{stage}: TextureMesh")

    cmd = [
        "TextureMesh",
        str(mesh_mvs),
    ]

    tool_runner.run(cmd, stage=stage + "_texture")

    if not textured_obj.exists():
        raise RuntimeError(f"{stage}: texture failed")

    logger.info(f"{stage}: SUCCESS → {textured_obj}")