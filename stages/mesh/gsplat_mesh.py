import json
import struct
import subprocess
from pathlib import Path
import numpy as np


def run_gsplat_mesh(paths, config, logger):
    logger.info("[GSPLAT] Starting GSplat pipeline")

    gs_cfg = config["mesh"]["gsplat"]

    # --------------------------------------------------
    # SETUP
    # --------------------------------------------------
    work_dir = paths.mesh / "gsplat"
    work_dir.mkdir(parents=True, exist_ok=True)

    transforms_path = work_dir / "transforms.json"
    model_dir = work_dir / "model"
    model_dir.mkdir(exist_ok=True)

    images_file = paths.sparse_model / "images.bin"
    images_dir = paths.images

    # --------------------------------------------------
    # STEP 1: COLMAP → transforms.json
    # --------------------------------------------------
    logger.info("[GSPLAT] Converting COLMAP → transforms.json")

    frames = []

    with open(images_file, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]

        for _ in range(num_images):
            f.read(4)

            qvec = struct.unpack("<4d", f.read(32))
            tvec = struct.unpack("<3d", f.read(24))

            f.read(4)

            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c

            name = name.decode()

            # skip points
            n_points = struct.unpack("<Q", f.read(8))[0]
            f.seek(n_points * 24, 1)

            # quaternion → rotation matrix
            q = np.array(qvec)
            q /= np.linalg.norm(q)

            w, x, y, z = q

            R = np.array([
                [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
                [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
                [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
            ])

            M = np.eye(4)
            M[:3, :3] = R
            M[:3, 3] = tvec

            frames.append({
                "file_path": str(images_dir / name),
                "transform_matrix": M.tolist()
            })

    with open(transforms_path, "w") as f:
        json.dump({"frames": frames}, f, indent=4)

    # --------------------------------------------------
    # STEP 2: TRAIN GSPLAT
    # --------------------------------------------------
    logger.info("[GSPLAT] Training GSplat model")

    train_cmd = [
        "python",
        "train.py",  # must exist in your gsplat repo
        "--data", str(transforms_path),
        "--output", str(model_dir),
        "--iterations", str(gs_cfg["iterations"]),
    ]

    subprocess.run(train_cmd, check=True)

    # --------------------------------------------------
    # STEP 3: EXTRACT MESH
    # --------------------------------------------------
    logger.info("[GSPLAT] Extracting mesh from trained model")

    extract_cmd = [
        "python",
        "extract_mesh.py",  # must exist or you implement it
        "--model_path", str(model_dir),
        "--output", str(paths.mesh_file),
        "--resolution", str(gs_cfg["extract_resolution"]),
        "--threshold", str(gs_cfg["density_thresh"]),
    ]

    subprocess.run(extract_cmd, check=True)

    logger.info(f"[GSPLAT] DONE → {paths.mesh_file}")