from pathlib import Path
import shutil


def build_workspace(paths, logger):
    workspace = paths.run_root / "mvs_workspace"

    if workspace.exists():
        shutil.rmtree(workspace)

    workspace.mkdir(parents=True)

    # Images
    src_images = (
        paths.images_downsampled
        if paths.images_downsampled.exists()
        else paths.images
    )

    shutil.copytree(src_images, workspace / "images")

    # Sparse
    if not paths.sparse_model.exists():
        raise RuntimeError("texture_mesh: sparse model missing")

    dst_sparse = workspace / "sparse" / "0"
    dst_sparse.parent.mkdir(parents=True)

    shutil.copytree(paths.sparse_model, dst_sparse)

    logger.info(f"texture_mesh: workspace ready")

    return workspace


def run(paths, config, logger, tool_runner):
    stage = "texture_mesh"
    logger.info(f"---- {stage.upper()} ----")

    workspace = build_workspace(paths, logger)

    mvs_dir = paths.texture
    mvs_dir.mkdir(parents=True, exist_ok=True)

    scene_mvs = mvs_dir / "scene.mvs"
    dense_mvs = mvs_dir / "scene_dense.mvs"
    mesh_mvs = mvs_dir / "scene_dense_mesh.mvs"
    textured_obj = mvs_dir / "scene_dense_mesh_texture.obj"

    # 🔥 CLEAN EVERYTHING (no stale reuse)
    for f in [scene_mvs, dense_mvs, mesh_mvs, textured_obj]:
        if f.exists():
            f.unlink()

    # =====================================================
    # 🔥 ADAPTIVE RESOLUTION LEVEL
    # =====================================================
    analysis = config.get("analysis_results", {})
    num_images = analysis.get("dataset", {}).get("num_images", 50)

    if num_images > 300:
        resolution = 2
    elif num_images > 100:
        resolution = 1
    else:
        resolution = 0  # 🔥 highest quality

    logger.info(f"{stage}: resolution_level={resolution}")

    # =====================================================
    # STEP 1: Interface
    # =====================================================
    tool_runner.run([
        "InterfaceCOLMAP",
        "-i", str(workspace),
        "-o", str(scene_mvs),
        "-w", str(mvs_dir),
    ], stage=stage + "_interface")

    # =====================================================
    # STEP 2: Densify
    # =====================================================
    tool_runner.run([
        "DensifyPointCloud",
        str(scene_mvs),
        "--resolution-level", str(resolution),
    ], stage=stage + "_densify")

    # =====================================================
    # STEP 3: Mesh
    # =====================================================
    tool_runner.run([
        "ReconstructMesh",
        str(dense_mvs),
    ], stage=stage + "_mesh")

    # =====================================================
    # STEP 4: Texture
    # =====================================================
    tool_runner.run([
        "TextureMesh",
        str(mesh_mvs),
    ], stage=stage + "_texture")

    if not textured_obj.exists():
        raise RuntimeError(f"{stage}: texture failed")

    logger.info(f"{stage}: SUCCESS → {textured_obj}")