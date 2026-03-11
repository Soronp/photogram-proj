#!/usr/bin/env python3
"""
gen_tex.py

MARK-2 Texture Generation Stage

Input
-----
mesh/mesh_cleaned.ply
openmvs/fused.mvs

Output
------
textures/textured_mesh.obj
textures/*
"""

from pathlib import Path
import shutil


# --------------------------------------------------
# Detect MVS scene
# --------------------------------------------------

def detect_scene(openmvs_root: Path):

    candidates = [

        "fused.mvs",
        "mesh_refined.mvs",
        "mesh.mvs",
        "scene.mvs"
    ]

    for name in candidates:

        p = openmvs_root / name

        if p.exists() and p.stat().st_size > 10000:
            return p

    raise RuntimeError("No valid MVS scene found")


# --------------------------------------------------
# Locate images
# --------------------------------------------------

def locate_images(openmvs_root: Path):

    candidates = [

        openmvs_root / "undistorted" / "images",
        openmvs_root / "images",
        openmvs_root / "undistorted"
    ]

    for c in candidates:

        if not c.exists():
            continue

        imgs = list(c.glob("*.jpg")) + list(c.glob("*.png"))

        if imgs:
            return c

    raise RuntimeError("Image directory not found")


# --------------------------------------------------
# Count images
# --------------------------------------------------

def count_images(image_dir: Path):

    patterns = ["*.jpg", "*.png", "*.JPG", "*.PNG"]

    total = 0

    for p in patterns:
        total += len(list(image_dir.glob(p)))

    return total


# --------------------------------------------------
# Resolution policy
# --------------------------------------------------

def choose_resolution(num_images):

    if num_images < 80:
        return 0

    if num_images < 300:
        return 1

    return 2


# --------------------------------------------------
# Run TextureMesh
# --------------------------------------------------

def run_texturemesh(tools, logger, scene, mesh, cwd, resolution, gpu=True):

    cmd = [

        "--input-file", scene.name,
        "--mesh-file", mesh.name,

        "--output-file", "textured_mesh.obj",

        "--export-type", "obj",

        "--resolution-level", str(resolution),

        "--texture-size", "8192",

        "--virtual-face-images", "3",

        "--verbosity", "2"
    ]

    cmd += ["--cuda-device", "0" if gpu else "-2"]

    logger.info(f"[gen_tex] running TextureMesh (GPU={'on' if gpu else 'off'})")

    tools.run(
        "openmvs.texturemesh",
        cmd,
        cwd=cwd
    )


# --------------------------------------------------
# Detect output mesh
# --------------------------------------------------

def detect_output(openmvs_root: Path):

    candidates = [

        "textured_mesh.obj",
        "model.obj",
        "scene.obj"
    ]

    for name in candidates:

        p = openmvs_root / name

        if p.exists():
            return p

    return None


# --------------------------------------------------
# Copy textures
# --------------------------------------------------

def copy_textures(openmvs_root: Path, textures_dir: Path):

    for obj in openmvs_root.glob("*.obj"):
        shutil.copy(obj, textures_dir / obj.name)

    for mtl in openmvs_root.glob("*.mtl"):

        shutil.copy(mtl, textures_dir / mtl.name)

        with open(mtl) as f:

            for line in f:

                if line.startswith("map_Kd"):

                    tex = line.strip().split()[-1]

                    tex_path = openmvs_root / tex

                    if tex_path.exists():
                        shutil.copy(tex_path, textures_dir / tex_path.name)


# --------------------------------------------------
# Stage entry
# --------------------------------------------------

def run(paths, tools, config, logger):

    logger.info("[gen_tex] starting")

    openmvs_root = paths.openmvs
    mesh_dir = paths.mesh
    textures_dir = paths.textures

    textures_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # Detect scene
    # --------------------------------------------------

    scene = detect_scene(openmvs_root)

    logger.info(f"[gen_tex] scene → {scene.name}")

    # --------------------------------------------------
    # Load mesh from mesh stage
    # --------------------------------------------------

    mesh = mesh_dir / "mesh_cleaned.ply"

    if not mesh.exists():
        raise RuntimeError("mesh_cleaned.ply missing")

    logger.info(f"[gen_tex] mesh → {mesh.name}")

    # --------------------------------------------------
    # Locate images
    # --------------------------------------------------

    image_dir = locate_images(openmvs_root)

    num_images = count_images(image_dir)

    resolution = choose_resolution(num_images)

    logger.info(f"[gen_tex] images detected: {num_images}")
    logger.info(f"[gen_tex] resolution level: {resolution}")

    # --------------------------------------------------
    # Copy mesh to OpenMVS workspace
    # --------------------------------------------------

    working_mesh = openmvs_root / mesh.name

    if working_mesh.exists():
        working_mesh.unlink()

    shutil.copy(mesh, working_mesh)

    # --------------------------------------------------
    # Run TextureMesh
    # --------------------------------------------------

    try:

        run_texturemesh(
            tools,
            logger,
            scene,
            working_mesh,
            openmvs_root,
            resolution,
            gpu=True
        )

    except Exception:

        logger.warning("[gen_tex] GPU failed — retrying CPU")

        run_texturemesh(
            tools,
            logger,
            scene,
            working_mesh,
            openmvs_root,
            resolution,
            gpu=False
        )

    # --------------------------------------------------
    # Detect output
    # --------------------------------------------------

    produced = detect_output(openmvs_root)

    if produced is None:
        raise RuntimeError("TextureMesh finished but produced no mesh")

    final_mesh = textures_dir / "textured_mesh.obj"

    produced.rename(final_mesh)

    # --------------------------------------------------
    # Copy textures
    # --------------------------------------------------

    copy_textures(openmvs_root, textures_dir)

    logger.info(f"[gen_tex] output → {final_mesh}")

    logger.info("[gen_tex] completed")