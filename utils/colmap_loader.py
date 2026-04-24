import struct
from pathlib import Path
from dataclasses import dataclass
from typing import Dict
import numpy as np


# =====================================================
# DATA STRUCTURES
# =====================================================

@dataclass
class Camera:
    id: int
    model: str
    width: int
    height: int
    params: np.ndarray


@dataclass
class Image:
    id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str
    xys: np.ndarray
    point3D_ids: np.ndarray


@dataclass
class Point3D:
    id: int
    xyz: np.ndarray
    rgb: np.ndarray
    error: float
    image_ids: np.ndarray
    point2D_idxs: np.ndarray


@dataclass
class ColmapModel:
    cameras: Dict[int, Camera]
    images: Dict[int, Image]
    points3D: Dict[int, Point3D]

    def sorted_images(self):
        return [self.images[k] for k in sorted(self.images)]

    def sorted_cameras(self):
        return [self.cameras[k] for k in sorted(self.cameras)]


# =====================================================
# CAMERA MODELS
# =====================================================

CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}


# =====================================================
# CAMERA LOADER
# =====================================================

def _read_cameras_binary(path: Path) -> Dict[int, Camera]:
    cameras = {}

    with open(path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]

        for _ in range(num):
            cam_id = struct.unpack("<I", f.read(4))[0]
            model_id = struct.unpack("<i", f.read(4))[0]
            width = struct.unpack("<Q", f.read(8))[0]
            height = struct.unpack("<Q", f.read(8))[0]

            model_name, n_params = CAMERA_MODELS[model_id]
            params = np.array(struct.unpack(f"<{n_params}d", f.read(8 * n_params)))

            cameras[cam_id] = Camera(
                cam_id, model_name, width, height, params
            )

    return cameras


# =====================================================
# IMAGE LOADER (FIXED - CRITICAL)
# =====================================================

def _read_images_binary(path: Path) -> Dict[int, Image]:
    images = {}

    with open(path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]

        for _ in range(num):
            img_id = struct.unpack("<I", f.read(4))[0]

            qvec = np.array(struct.unpack("<4d", f.read(32)))
            tvec = np.array(struct.unpack("<3d", f.read(24)))

            cam_id = struct.unpack("<I", f.read(4))[0]

            # read null-terminated string
            name_bytes = bytearray()
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name_bytes.extend(c)
            name = name_bytes.decode("utf-8")

            num_points = struct.unpack("<Q", f.read(8))[0]

            xys = np.zeros((num_points, 2), dtype=np.float64)
            point_ids = np.full((num_points,), -1, dtype=np.int64)

            for i in range(num_points):
                x, y = struct.unpack("<dd", f.read(16))
                pid = struct.unpack("<q", f.read(8))[0]

                xys[i] = [x, y]
                point_ids[i] = pid  # can be -1

            images[img_id] = Image(
                id=img_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=cam_id,
                name=name,
                xys=xys,
                point3D_ids=point_ids,
            )

    return images


# =====================================================
# POINTS LOADER
# =====================================================

def _read_points3D_binary(path: Path) -> Dict[int, Point3D]:
    points = {}

    with open(path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]

        for _ in range(num):
            pid = struct.unpack("<Q", f.read(8))[0]

            xyz = np.array(struct.unpack("<3d", f.read(24)))
            rgb = np.array(struct.unpack("<3B", f.read(3)))
            error = struct.unpack("<d", f.read(8))[0]

            track_len = struct.unpack("<Q", f.read(8))[0]

            image_ids = np.zeros(track_len, dtype=np.uint32)
            point_idxs = np.zeros(track_len, dtype=np.uint32)

            for i in range(track_len):
                iid, pidx = struct.unpack("<II", f.read(8))
                image_ids[i] = iid
                point_idxs[i] = pidx

            points[pid] = Point3D(
                pid, xyz, rgb, error, image_ids, point_idxs
            )

    return points


# =====================================================
# VALIDATION
# =====================================================

def _validate_model(cameras, images):
    if not images:
        raise RuntimeError("No images found in COLMAP model")

    for img in images.values():
        if img.camera_id not in cameras:
            raise RuntimeError(
                f"Image {img.id} references missing camera {img.camera_id}"
            )


# =====================================================
# MAIN ENTRY
# =====================================================

def load_colmap_model(sparse_path: Path) -> ColmapModel:
    cams_path = sparse_path / "cameras.bin"
    imgs_path = sparse_path / "images.bin"
    pts_path = sparse_path / "points3D.bin"

    if not cams_path.exists():
        raise FileNotFoundError(f"Missing {cams_path}")
    if not imgs_path.exists():
        raise FileNotFoundError(f"Missing {imgs_path}")
    if not pts_path.exists():
        raise FileNotFoundError(f"Missing {pts_path}")

    cameras = _read_cameras_binary(cams_path)
    images = _read_images_binary(imgs_path)
    points = _read_points3D_binary(pts_path)

    _validate_model(cameras, images)

    return ColmapModel(cameras, images, points)