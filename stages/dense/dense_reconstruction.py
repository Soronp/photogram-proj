#!/usr/bin/env python3
"""
dense_reconstruction.py

MARK-2 Dense Reconstruction Stage

Input
-----
openmvs/scene.mvs

Output
------
openmvs/fused.mvs
openmvs/fused.ply
dense/fused.ply
"""

from pathlib import Path
import shutil


# --------------------------------------------------
# Validate scene
# --------------------------------------------------

def validate_scene(scene: Path):

    if not scene.exists():
        raise RuntimeError("scene.mvs missing")

    if scene.stat().st_size < 1000:
        raise RuntimeError("scene.mvs appears corrupt")


# --------------------------------------------------
# Count undistorted images
# --------------------------------------------------

def count_images(image_dir: Path):

    imgs = (
        list(image_dir.glob("*.jpg"))
        + list(image_dir.glob("*.png"))
        + list(image_dir.glob("*.JPG"))
        + list(image_dir.glob("*.PNG"))
    )

    if not imgs:
        raise RuntimeError("No undistorted images found")

    return len(imgs)


# --------------------------------------------------
# Parameter selection
# --------------------------------------------------

def compute_params(image_count):
    """
    Choose OpenMVS parameters based on dataset size.

    Small datasets need higher detail
    because fewer views exist.
    """

    if image_count <= 40:

        return dict(
            resolution_level=0,
            views=5,
            max_resolution=4000
        )

    if image_count <= 150:

        return dict(
            resolution_level=1,
            views=4,
            max_resolution=3500
        )

    if image_count <= 600:

        return dict(
            resolution_level=1,
            views=4,
            max_resolution=3000
        )

    return dict(
        resolution_level=2,
        views=3,
        max_resolution=2500
    )


# --------------------------------------------------
# Run densify
# --------------------------------------------------

def run_densify(tools, logger, root, params, gpu=True):

    cmd = [

        "--input-file", "scene.mvs",

        "--output-file", "fused.mvs",

        "--resolution-level", str(params["resolution_level"]),

        "--max-resolution", str(params["max_resolution"]),

        "--number-views", str(params["views"]),

        "--fusion-mode", "1",

        "--estimate-colors", "2",
        "--estimate-normals", "2",

        "--verbosity", "3"
    ]

    if gpu:
        cmd += ["--cuda-device", "0"]
    else:
        cmd += ["--cuda-device", "-2"]

    logger.info(
        f"[dense] DensifyPointCloud GPU={'on' if gpu else 'off'}"
    )

    tools.run(
        "openmvs.densifypointcloud",
        cmd,
        cwd=root
    )


# --------------------------------------------------
# Validate dense cloud
# --------------------------------------------------

def validate_dense_cloud(ply_path: Path):

    if not ply_path.exists():
        raise RuntimeError("Dense cloud not produced")

    with open(ply_path, "rb") as f:
        header = f.read(2000).decode(errors="ignore")

    for line in header.splitlines():

        if "element vertex" in line:

            count = int(line.split()[-1])

            if count < 10000:
                raise RuntimeError(
                    f"Dense cloud too small ({count} points)"
                )

            return count

    raise RuntimeError("Unable to read PLY header")


# --------------------------------------------------
# Stage entry
# --------------------------------------------------

def run(paths, logger, tools, config):

    logger.info("[dense] stage start")

    openmvs_root = paths.openmvs
    dense_dir = paths.dense

    dense_dir.mkdir(parents=True, exist_ok=True)

    scene = openmvs_root / "scene.mvs"

    validate_scene(scene)

    # --------------------------------------------------
    # Count images
    # --------------------------------------------------

    image_dir = openmvs_root / "undistorted" / "images"

    if not image_dir.exists():
        raise RuntimeError("Undistorted images missing")

    image_count = count_images(image_dir)

    logger.info(f"[dense] images detected → {image_count}")

    params = compute_params(image_count)

    logger.info(f"[dense] params → {params}")

    fused_mvs = openmvs_root / "fused.mvs"
    fused_ply = openmvs_root / "fused.ply"

    final_cloud = dense_dir / "fused.ply"

    # --------------------------------------------------
    # Skip stage if already done
    # --------------------------------------------------

    if fused_mvs.exists() and final_cloud.exists():

        logger.info("[dense] outputs already exist — skipping")

        return

    # --------------------------------------------------
    # Run densify
    # --------------------------------------------------

    try:

        run_densify(
            tools,
            logger,
            openmvs_root,
            params,
            gpu=True
        )

    except Exception:

        logger.warning("[dense] GPU failed — retrying CPU")

        run_densify(
            tools,
            logger,
            openmvs_root,
            params,
            gpu=False
        )

    if not fused_mvs.exists():
        raise RuntimeError("OpenMVS densify failed")

    logger.info("[dense] densify completed")

    # --------------------------------------------------
    # Validate dense cloud
    # --------------------------------------------------

    point_count = validate_dense_cloud(fused_ply)

    logger.info(f"[dense] dense points → {point_count}")

    # --------------------------------------------------
    # Copy result to pipeline output
    # --------------------------------------------------

    shutil.copy2(fused_ply, final_cloud)

    logger.info(f"[dense] output → {final_cloud}")

    logger.info("[dense] stage completed")