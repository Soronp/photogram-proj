from pathlib import Path


def run(paths, config, logger, tool_runner):
    stage = "openmvs_export"
    logger.info(f"---- {stage.upper()} ----")

    sparse_model = paths.sparse_model
    image_dir = paths.images

    if not sparse_model.exists():
        raise RuntimeError(f"{stage}: sparse model missing")

    if not image_dir.exists():
        raise RuntimeError(f"{stage}: image directory missing")

    # =====================================================
    # OUTPUT DIR (OpenMVS workspace)
    # =====================================================
    mvs_dir = paths.run_root / "openmvs"
    mvs_dir.mkdir(parents=True, exist_ok=True)

    scene_file = mvs_dir / "scene.mvs"

    # Clean old scene
    if scene_file.exists():
        logger.warning(f"{stage}: removing old scene.mvs")
        scene_file.unlink()

    logger.info(f"{stage}: exporting COLMAP → OpenMVS")

    # =====================================================
    # 🔥 CRITICAL: CORRECT PATH USAGE
    # =====================================================
    cmd = [
        "InterfaceCOLMAP",

        # Input: COLMAP project root (NOT sparse/0)
        "-i", str(paths.run_root),

        # Output scene
        "-o", str(scene_file),

        # Working dir
        "-w", str(mvs_dir),
    ]

    tool_runner.run(cmd, stage=stage)

    # =====================================================
    # VALIDATION
    # =====================================================
    if not scene_file.exists():
        raise RuntimeError(f"{stage}: scene.mvs not created")

    logger.info(f"{stage}: SUCCESS → {scene_file}")