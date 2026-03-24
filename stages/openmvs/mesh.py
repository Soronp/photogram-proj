from pathlib import Path


def run(paths, config, logger, tool_runner):
    stage = "openmvs_mesh"
    logger.info(f"---- {stage.upper()} ----")

    # =====================================================
    # 📁 WORKSPACE
    # =====================================================
    mvs_dir = (paths.run_root / "openmvs").resolve()

    dense_scene = mvs_dir / "scene_dense.mvs"

    if not dense_scene.exists():
        raise RuntimeError(f"{stage}: missing scene_dense.mvs → run densify first")

    # =====================================================
    # 🎯 OUTPUTS
    # =====================================================
    paths.mesh.mkdir(parents=True, exist_ok=True)

    mesh_ply = paths.mesh_file.resolve()
    mesh_obj = paths.mesh / "mesh.obj"

    # =====================================================
    # ⚙️ CONFIG
    # =====================================================
    cfg = config.get("mesh", {})

    cuda_device = cfg.get("cuda_device", 0)
    free_space_support = cfg.get("free_space_support", 0)
    remove_spikes = cfg.get("remove_spikes", 1)
    close_holes = cfg.get("close_holes", 30)

    # =====================================================
    # 🚀 COMMAND (FIXED PARITY WITH CLI)
    # =====================================================
    cmd = [
        "ReconstructMesh",

        # INPUT (ONLY THIS IS CORRECT FOR YOUR CASE)
        "-i", str(dense_scene),

        # OUTPUT
        "-o", str(mesh_ply),

        # WORKSPACE
        "-w", str(mvs_dir),

        # QUALITY CONTROL
        "-f", str(free_space_support),
        "--remove-spikes", str(remove_spikes),
        "--close-holes", str(close_holes),

        # GPU CONTROL
        "--cuda-device", str(cuda_device),
    ]

    logger.info(f"[{stage}] COMMAND:")
    logger.info(" ".join(cmd))

    # =====================================================
    # ▶️ EXECUTION
    # =====================================================
    result = tool_runner.run(cmd, stage=stage)

    # =====================================================
    # 🔍 REAL FAILURE DETECTION
    # =====================================================
    if not mesh_ply.exists() and not mesh_obj.exists():
        raise RuntimeError(
            f"{stage}: mesh failed → no output generated (check densify point cloud validity)"
        )

    # =====================================================
    # 🎯 OUTPUT RESOLUTION
    # =====================================================
    final_mesh = mesh_ply if mesh_ply.exists() else mesh_obj

    logger.info(f"{stage}: SUCCESS → {final_mesh}")