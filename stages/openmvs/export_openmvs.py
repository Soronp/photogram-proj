from pathlib import Path
import shutil


# =====================================================
# 🔍 VALIDATE DENSE COLMAP WORKSPACE
# =====================================================

def _validate_dense_workspace(dense_dir: Path):
    sparse_dir = dense_dir / "sparse"
    image_dir = dense_dir / "images"

    required = ["cameras.bin", "images.bin", "points3D.bin"]

    if not sparse_dir.exists():
        raise RuntimeError("openmvs_export: missing dense/sparse")

    if not image_dir.exists():
        raise RuntimeError("openmvs_export: missing dense/images")

    missing = [f for f in required if not (sparse_dir / f).exists()]
    if missing:
        raise RuntimeError(
            f"openmvs_export: invalid dense sparse model → missing {missing}"
        )

    images = [p for p in image_dir.glob("*") if p.is_file()]
    if not images:
        raise RuntimeError("openmvs_export: no images found in dense/images")

    return sparse_dir, image_dir, images


# =====================================================
# 🏗️ BUILD OPENMVS WORKSPACE (CRITICAL FIX)
# =====================================================

def _build_openmvs_workspace(paths, sparse_dir, image_dir, images, logger):
    workspace = (paths.run_root / "openmvs_workspace").resolve()
    ws_images = workspace / "images"
    ws_sparse = workspace / "sparse"

    # 🔥 HARD RESET (prevents stale corruption)
    if workspace.exists():
        logger.warning("openmvs_export: clearing previous workspace")
        shutil.rmtree(workspace)

    ws_images.mkdir(parents=True, exist_ok=True)
    ws_sparse.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------
    # 📸 COPY IMAGES (UNDISTORTED)
    # -------------------------------------------------
    logger.info("openmvs_export: copying undistorted images")

    for img in images:
        shutil.copy2(img, ws_images / img.name)

    # -------------------------------------------------
    # 📦 COPY SPARSE MODEL (FLATTENED)
    # -------------------------------------------------
    logger.info("openmvs_export: copying sparse model")

    for fname in ["cameras.bin", "images.bin", "points3D.bin"]:
        src = sparse_dir / fname
        dst = ws_sparse / fname

        if not src.exists():
            raise RuntimeError(f"openmvs_export: missing {src}")

        shutil.copy2(src, dst)

    # -------------------------------------------------
    # ✅ FINAL STRUCTURE CHECK
    # -------------------------------------------------
    if not ws_images.exists() or not ws_sparse.exists():
        raise RuntimeError("openmvs_export: workspace build failed")

    logger.info(f"openmvs_export: workspace ready → {workspace}")

    return workspace


# =====================================================
# 🚀 MAIN EXPORT (FULLY FIXED)
# =====================================================

def run(paths, config, logger, tool_runner):
    stage = "openmvs_export"
    logger.info(f"---- {stage.upper()} ----")

    # =====================================================
    # 1. VALIDATE INPUT (COLMAP DENSE OUTPUT)
    # =====================================================
    dense_dir = paths.dense.resolve()

    sparse_dir, image_dir, images = _validate_dense_workspace(dense_dir)

    logger.info(f"{stage}: using dense sparse → {sparse_dir}")
    logger.info(f"{stage}: using dense images → {image_dir}")
    logger.info(f"{stage}: image count = {len(images)}")

    # =====================================================
    # 2. BUILD CLEAN OPENMVS WORKSPACE
    # =====================================================
    workspace = _build_openmvs_workspace(
        paths, sparse_dir, image_dir, images, logger
    )

    # =====================================================
    # 3. OPENMVS OUTPUT DIR
    # =====================================================
    mvs_dir = (paths.run_root / "openmvs").resolve()
    mvs_dir.mkdir(parents=True, exist_ok=True)

    scene_file = mvs_dir / "scene.mvs"

    if scene_file.exists():
        logger.warning(f"{stage}: removing old scene.mvs")
        scene_file.unlink()

    # =====================================================
    # 4. COMMAND (STRICT WORKSPACE MODE)
    # =====================================================
    cmd = [
        "InterfaceCOLMAP",

        # 🔥 CRITICAL: pass workspace ROOT (not sparse!)
        "-i", str(workspace),

        # output
        "-o", str(scene_file),

        # OpenMVS working dir
        "-w", str(mvs_dir),
    ]

    logger.info(f"[{stage}] COMMAND:\n{' '.join(cmd)}")

    # =====================================================
    # 5. EXECUTION
    # =====================================================
    tool_runner.run(cmd, stage=stage)

    # =====================================================
    # 6. VALIDATION
    # =====================================================
    if not scene_file.exists():
        raise RuntimeError(f"{stage}: scene.mvs not created")

    logger.info(f"{stage}: SUCCESS → {scene_file}")