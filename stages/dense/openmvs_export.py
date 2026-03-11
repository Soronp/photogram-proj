#!/usr/bin/env python3
"""
openmvs_export.py

MARK-2 Pipeline Stage
Convert COLMAP sparse reconstruction into OpenMVS scene.

Pipeline
--------
COLMAP sparse
      ↓
image_undistorter
      ↓
openmvs/undistorted/
      ├── images/
      └── sparse/
      ↓
InterfaceCOLMAP
      ↓
openmvs/scene.mvs
"""

import shutil
from pathlib import Path


REQUIRED_SPARSE = {
    "cameras.bin",
    "images.bin",
    "points3D.bin"
}


# --------------------------------------------------
# Find valid sparse model
# --------------------------------------------------

def find_sparse_model(sparse_root: Path):

    models = []

    for d in sparse_root.iterdir():

        if not d.is_dir():
            continue

        files = {p.name for p in d.iterdir()}

        if REQUIRED_SPARSE.issubset(files):

            points_file = d / "points3D.bin"

            size = points_file.stat().st_size

            models.append((size, d))

    if not models:
        raise RuntimeError("No valid COLMAP sparse model found")

    # choose model with most points
    models.sort(reverse=True)

    return models[0][1]


# --------------------------------------------------
# Count images helper
# --------------------------------------------------

def count_images(folder: Path):

    exts = ["*.jpg", "*.png", "*.JPG", "*.PNG"]

    total = 0

    for e in exts:
        total += len(list(folder.glob(e)))

    return total


# --------------------------------------------------
# Stage entry
# --------------------------------------------------

def run(paths, logger, tools, config):

    logger.info("[openmvs_export] starting")

    openmvs_root = paths.openmvs
    openmvs_root.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # Locate sparse reconstruction
    # --------------------------------------------------

    sparse_model = find_sparse_model(paths.sparse)

    logger.info(f"[openmvs_export] sparse model → {sparse_model}")

    # --------------------------------------------------
    # Locate filtered images
    # --------------------------------------------------

    image_src = paths.images_filtered

    if not image_src.exists():
        raise RuntimeError("Filtered images directory missing")

    logger.info(f"[openmvs_export] images → {image_src}")

    # --------------------------------------------------
    # Run COLMAP undistortion
    # --------------------------------------------------

    undistorted_root = openmvs_root / "undistorted"

    sparse_out = undistorted_root / "sparse"
    images_out = undistorted_root / "images"

    if not sparse_out.exists():

        logger.info("[openmvs_export] running image_undistorter")

        tools.run(
            "colmap",
            [
                "image_undistorter",
                "--image_path", str(image_src),
                "--input_path", str(sparse_model),
                "--output_path", str(undistorted_root),
                "--output_type", "COLMAP",
            ]
        )

        if not sparse_out.exists():
            raise RuntimeError("COLMAP undistortion failed")

    # --------------------------------------------------
    # Validate undistorted images
    # --------------------------------------------------

    if not images_out.exists():
        raise RuntimeError("Undistorted images missing")

    image_count = count_images(images_out)

    if image_count == 0:
        raise RuntimeError("No undistorted images produced")

    logger.info(f"[openmvs_export] undistorted images → {image_count}")

    # --------------------------------------------------
    # Remove COLMAP stereo directory (conflicts with OpenMVS)
    # --------------------------------------------------

    stereo_dir = undistorted_root / "stereo"

    if stereo_dir.exists():

        logger.info("[openmvs_export] removing stereo directory")

        shutil.rmtree(stereo_dir)

    # --------------------------------------------------
    # Run OpenMVS interface
    # --------------------------------------------------

    scene_file = openmvs_root / "scene.mvs"

    if scene_file.exists():

        logger.info("[openmvs_export] scene already exists — skipping")

        return

    logger.info("[openmvs_export] running InterfaceCOLMAP")

    tools.run(
        "openmvs.interface",
        [
            "-i", "undistorted",
            "-o", "scene.mvs"
        ],
        cwd=openmvs_root
    )

    if not scene_file.exists():
        raise RuntimeError("OpenMVS scene.mvs not produced")

    logger.info(f"[openmvs_export] scene created → {scene_file}")

    logger.info("[openmvs_export] completed")