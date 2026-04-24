import json
import shutil
from pathlib import Path
import numpy as np

from utils.colmap_loader import load_colmap_model


# =====================================================
# POSE UTILS
# =====================================================

def qvec_to_rotmat(qvec):
    qvec = qvec / np.linalg.norm(qvec)
    w, x, y, z = qvec

    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])


def build_c2w(qvec, tvec):
    R = qvec_to_rotmat(qvec)
    t = tvec.reshape(3, 1)

    w2c = np.eye(4)
    w2c[:3, :3] = R
    w2c[:3, 3:] = t

    return np.linalg.inv(w2c)


def convert_to_opengl(c2w):
    c2w = c2w.copy()
    c2w[0:3, 1:3] *= -1
    return c2w


# =====================================================
# INTRINSICS
# =====================================================

def extract_intrinsics(camera):
    model = camera.model
    p = camera.params

    if model in ["SIMPLE_PINHOLE", "SIMPLE_RADIAL"]:
        fx = fy = p[0]
        cx, cy = p[1], p[2]
        k1 = p[3] if len(p) > 3 else 0.0
        k2 = 0.0

    elif model in ["PINHOLE"]:
        fx, fy = p[0], p[1]
        cx, cy = p[2], p[3]
        k1 = k2 = 0.0

    elif model in ["OPENCV", "FULL_OPENCV"]:
        fx, fy = p[0], p[1]
        cx, cy = p[2], p[3]
        k1, k2 = p[4], p[5]

    else:
        raise ValueError(f"Unsupported camera model: {model}")

    return fx, fy, cx, cy, k1, k2


# =====================================================
# NORMALIZATION
# =====================================================

def normalize_poses(poses):
    centers = poses[:, :3, 3]
    center = centers.mean(axis=0)
    scale = 1.0 / np.max(np.linalg.norm(centers - center, axis=1))
    return center, scale


# =====================================================
# IMAGE RESOLUTION FIX (CRITICAL)
# =====================================================

def resolve_image_path(image_root: Path, colmap_name: str):
    """
    Try multiple strategies to find the image.
    """
    candidates = [
        image_root / colmap_name,
        image_root / Path(colmap_name).name,
        image_root / "images" / Path(colmap_name).name,
    ]

    for c in candidates:
        if c.exists():
            return c

    return None


# =====================================================
# MAIN
# =====================================================

def run_colmap_to_nerfstudio(paths, config, logger):

    logger.info("==== COLMAP → NERFSTUDIO (ROBUST) ====")

    model = load_colmap_model(paths.sparse_model)

    images = model.sorted_images()
    cameras = {cam.id: cam for cam in model.cameras.values()}

    dataset_dir = paths.dense / "nerfstudio_dataset"
    images_out = dataset_dir / "images"

    dataset_dir.mkdir(parents=True, exist_ok=True)
    images_out.mkdir(parents=True, exist_ok=True)

    frames = []
    poses = []

    total = len(images)
    kept = 0

    for img in images:

        src = resolve_image_path(paths.images, img.name)

        if src is None:
            logger.warning(f"[NERF] Missing image: {img.name}")
            continue

        dst = images_out / Path(img.name).name

        if not dst.exists():
            shutil.copy(src, dst)

        cam = cameras[img.camera_id]
        fx, fy, cx, cy, k1, k2 = extract_intrinsics(cam)

        c2w = convert_to_opengl(build_c2w(img.qvec, img.tvec))

        poses.append(c2w)

        frames.append({
            "file_path": f"images/{dst.name}",
            "transform_matrix": c2w.tolist()
        })

        kept += 1

    logger.info(f"[NERF] Frames kept: {kept}/{total}")

    # 🚨 HARD FAIL if too many dropped
    if kept < 0.7 * total:
        raise RuntimeError(
            f"[NERF] Too many images missing: {kept}/{total}"
        )

    poses = np.stack(poses)

    if config["dense"]["nerfstudio"].get("normalize", True):
        center, scale = normalize_poses(poses)

        for f in frames:
            M = np.array(f["transform_matrix"])
            M[:3, 3] = (M[:3, 3] - center) * scale
            f["transform_matrix"] = M.tolist()

    # Use first camera ONLY for global metadata (Nerfstudio requirement)
    cam0 = cameras[images[0].camera_id]
    fx, fy, cx, cy, k1, k2 = extract_intrinsics(cam0)

    transforms = {
        "camera_model": "OPENCV",
        "fl_x": fx,
        "fl_y": fy,
        "cx": cx,
        "cy": cy,
        "k1": k1,
        "k2": k2,
        "p1": 0.0,
        "p2": 0.0,
        "w": int(cam0.width),
        "h": int(cam0.height),
        "frames": frames
    }

    out_path = dataset_dir / "transforms.json"
    out_path.write_text(json.dumps(transforms, indent=4))

    logger.info(f"[NERF] Dataset ready → {dataset_dir}")

    return dataset_dir