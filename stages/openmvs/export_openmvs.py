from pathlib import Path
import shutil

# =====================================================
# VALIDATE COLMAP DENSE WORKSPACE
# =====================================================
def _validate_colmap_dense(dense_dir: Path):
    sparse_dir = dense_dir / "sparse"
    image_dir = dense_dir / "images"
    required = ["cameras.bin", "images.bin", "points3D.bin"]

    if not sparse_dir.exists():
        raise RuntimeError("openmvs_export: missing dense/sparse")
    if not image_dir.exists():
        raise RuntimeError("openmvs_export: missing dense/images")

    missing = [f for f in required if not (sparse_dir / f).exists()]
    if missing:
        raise RuntimeError(f"openmvs_export: missing {missing}")

    images = [p for p in image_dir.glob("*") if p.is_file()]
    if not images:
        raise RuntimeError("openmvs_export: no images found")

    return sparse_dir, image_dir, images

# =====================================================
# VALIDATE OPENMVG OUTPUT
# =====================================================
def _validate_openmvg(paths):
    reconstruction_dir = paths.sparse / "openmvg_reconstruction"
    sfm_file = reconstruction_dir / "sfm_data.bin"

    if not sfm_file.exists():
        raise RuntimeError("openmvs_export: missing OpenMVG sfm_data.bin")

    image_dir = paths.images
    images = [p for p in image_dir.glob("*") if p.is_file()]
    if not images:
        raise RuntimeError("openmvs_export: no images found for OpenMVG")

    return sfm_file, image_dir, images

# =====================================================
# BUILD OPENMVS WORKSPACE (COLMAP-style)
# =====================================================
def _build_workspace(paths, images, sparse_files=None, logger=None):
    workspace = (paths.run_root / "openmvs_workspace").resolve()
    ws_images = workspace / "images"
    ws_sparse = workspace / "sparse"

    if workspace.exists():
        if logger:
            logger.warning("openmvs_export: clearing previous workspace")
        shutil.rmtree(workspace)

    ws_images.mkdir(parents=True, exist_ok=True)
    ws_sparse.mkdir(parents=True, exist_ok=True)

    if logger:
        logger.info("openmvs_export: copying images")
    for img in images:
        shutil.copy2(img, ws_images / img.name)

    if sparse_files:
        if logger:
            logger.info("openmvs_export: copying sparse files")
        for fname in sparse_files:
            shutil.copy2(sparse_files[fname], ws_sparse / fname)

    return workspace

# =====================================================
# MAIN EXPORT (MULTI-BACKEND)
# =====================================================
def run(paths, config, logger, tool_runner):
    stage = "openmvs_export"
    logger.info(f"---- {stage.upper()} ----")

    sparse_backend = config["pipeline"]["backends"]["sparse"]

    mvs_dir = (paths.run_root / "openmvs").resolve()
    mvs_dir.mkdir(parents=True, exist_ok=True)

    scene_file = mvs_dir / "scene.mvs"
    if scene_file.exists():
        logger.warning(f"{stage}: removing old scene.mvs")
        scene_file.unlink()

    # =====================================================
    # COLMAP PIPELINE
    # =====================================================
    if sparse_backend == "colmap":
        dense_dir = paths.dense.resolve()
        sparse_dir, image_dir, images = _validate_colmap_dense(dense_dir)
        logger.info(f"{stage}: COLMAP mode, images = {len(images)}")

        sparse_files = {f: (sparse_dir / f) for f in ["cameras.bin", "images.bin", "points3D.bin"]}
        workspace = _build_workspace(paths, images, sparse_files, logger)

        cmd = [
            "InterfaceCOLMAP",
            "-i", str(workspace),
            "-o", str(scene_file),
            "-w", str(mvs_dir),
        ]

    # =====================================================
    # OPENMVG PIPELINE WITH UNDISTORTION
    # =====================================================
    elif sparse_backend == "openmvg":
        sfm_file, image_dir, images = _validate_openmvg(paths)
        logger.info(f"{stage}: OpenMVG mode, images = {len(images)}")
        logger.info(f"{stage}: sfm_data → {sfm_file}")

        # Step 1: Undistort images
        undistorted_dir = paths.run_root / "undistorted_images"
        undistorted_dir.mkdir(exist_ok=True)

        cmd_undistort = [
            "openMVG_main_ExportUndistortedImages",
            "-i", str(sfm_file),
            "-o", str(undistorted_dir)
        ]
        logger.info(f"[{stage}] UNDISTORT COMMAND:\n{' '.join(cmd_undistort)}")
        tool_runner.run(cmd_undistort, stage=f"{stage}_undistort")

        undistorted_images = list(undistorted_dir.glob("*"))

        # Step 2: Build workspace using undistorted images
        workspace = _build_workspace(paths, undistorted_images, sparse_files={"sfm_data.bin": sfm_file}, logger=logger)

        # Step 3: Convert to OpenMVS format
        cmd = [
            "openMVG_main_openMVG2openMVS",
            "-i", str(sfm_file),
            "-o", str(scene_file),
            "-d", str(undistorted_dir),
        ]

    else:
        raise ValueError(f"{stage}: unsupported sparse backend {sparse_backend}")

    # =====================================================
    # 🚀 EXECUTE
    # =====================================================
    logger.info(f"[{stage}] COMMAND:\n{' '.join(cmd)}")
    tool_runner.run(cmd, stage=stage)

    # =====================================================
    # VALIDATION
    # =====================================================
    if not scene_file.exists():
        raise RuntimeError(f"{stage}: scene.mvs not created")

    logger.info(f"{stage}: SUCCESS → {scene_file}")