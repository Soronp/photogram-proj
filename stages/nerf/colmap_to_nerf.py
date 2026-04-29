from pathlib import Path
import json
import shutil
import numpy as np

from utils.colmap_loader import load_colmap_model


# =====================================================
# POSE UTILS (STRICT COLMAP)
# =====================================================

def qvec_to_rotmat(qvec):
    qvec = qvec / np.linalg.norm(qvec)
    w, x, y, z = qvec

    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ], dtype=np.float64)


def build_c2w(qvec, tvec):
    """
    Convert COLMAP world-to-camera → camera-to-world
    WITHOUT modifying coordinate system
    """
    R = qvec_to_rotmat(qvec)
    t = tvec.reshape(3, 1)

    w2c = np.eye(4)
    w2c[:3, :3] = R
    w2c[:3, 3:] = t

    return np.linalg.inv(w2c)


# =====================================================
# INTRINSICS (STRICT PER-CAMERA)
# =====================================================

def extract_intrinsics(camera):
    model = camera.model
    p = camera.params

    if model in ["SIMPLE_PINHOLE", "SIMPLE_RADIAL"]:
        fx = fy = p[0]
        cx, cy = p[1], p[2]
        k1 = p[3] if len(p) > 3 else 0.0
        k2 = 0.0

    elif model == "PINHOLE":
        fx, fy = p[0], p[1]
        cx, cy = p[2], p[3]
        k1 = k2 = 0.0

    elif model in ["OPENCV", "FULL_OPENCV"]:
        fx, fy = p[0], p[1]
        cx, cy = p[2], p[3]
        k1, k2 = p[4], p[5]

    else:
        raise ValueError(f"[NERF] Unsupported camera model: {model}")

    return float(fx), float(fy), float(cx), float(cy), float(k1), float(k2)


# =====================================================
# IMAGE RESOLUTION
# =====================================================

def resolve_image_path(image_root: Path, name: str):
    candidates = [
        image_root / name,
        image_root / Path(name).name,
        image_root / "images" / Path(name).name,
    ]

    for c in candidates:
        if c.exists():
            return c

    return None


# =====================================================
# MAIN
# =====================================================

def run_colmap_to_nerfstudio(paths, config, logger):

    logger.info("==== COLMAP → NERFSTUDIO (ACCURACY MODE) ====")

    model = load_colmap_model(paths.sparse_model)

    images = model.sorted_images()
    cameras = {cam.id: cam for cam in model.cameras.values()}

    if not images:
        raise RuntimeError("[NERF] No images found in COLMAP model")

    dataset_dir = paths.dense / "nerfstudio_dataset"
    images_out = dataset_dir / "images"

    dataset_dir.mkdir(parents=True, exist_ok=True)
    images_out.mkdir(parents=True, exist_ok=True)

    frames = []
    total = len(images)
    kept = 0

    # =====================================================
    # BUILD FRAMES (NO TRANSFORMATIONS)
    # =====================================================

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

        c2w = build_c2w(img.qvec, img.tvec)

        frames.append({
            "file_path": f"images/{dst.name}",
            "transform_matrix": c2w.tolist(),

            # 🔥 PER-FRAME INTRINSICS (CRITICAL)
            "fl_x": fx,
            "fl_y": fy,
            "cx": cx,
            "cy": cy,
        })

        kept += 1

    logger.info(f"[NERF] Frames kept: {kept}/{total}")

    if kept < 0.7 * total:
        raise RuntimeError(f"[NERF] Too many missing images: {kept}/{total}")

    # =====================================================
    # GLOBAL CAMERA (ONLY FOR DEFAULTS)
    # =====================================================

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

    # Debug
    logger.info("[NERF DEBUG] First pose:")
    logger.info(np.array(frames[0]["transform_matrix"]))

    return dataset_dir