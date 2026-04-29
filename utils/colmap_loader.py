import struct
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List
import numpy as np


# =====================================================
# NUMERICAL UTILS
# =====================================================

def _normalize_quaternion(q):
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm < 1e-10:
        raise RuntimeError("Invalid quaternion (near zero norm)")
    return q / norm


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

    # -------------------------------------------------
    # Rotation (STABLE)
    # -------------------------------------------------
    def rotmat(self) -> np.ndarray:
        q = _normalize_quaternion(self.qvec)
        w, x, y, z = q

        return np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
        ], dtype=np.float64)

    # -------------------------------------------------
    # Pose (COLMAP → c2w)
    # -------------------------------------------------
    def c2w(self) -> np.ndarray:
        R = self.rotmat()
        t = self.tvec.reshape(3, 1)

        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :3] = R
        w2c[:3, 3:] = t

        return np.linalg.inv(w2c)

    def camera_center(self) -> np.ndarray:
        return self.c2w()[:3, 3]


@dataclass
class Point3D:
    id: int
    xyz: np.ndarray
    rgb: np.ndarray
    error: float
    image_ids: np.ndarray
    point2D_idxs: np.ndarray

    def track_length(self) -> int:
        return len(self.image_ids)


@dataclass
class ColmapModel:
    cameras: Dict[int, Camera]
    images: Dict[int, Image]
    points3D: Dict[int, Point3D]

    # -------------------------------------------------
    # Accessors
    # -------------------------------------------------
    def sorted_images(self) -> List[Image]:
        return [self.images[k] for k in sorted(self.images)]

    def sorted_cameras(self) -> List[Camera]:
        return [self.cameras[k] for k in sorted(self.cameras)]

    # -------------------------------------------------
    # Quality Filtering (CRITICAL for stability)
    # -------------------------------------------------
    def filter_points(self, min_track=3, max_error=2.0):
        return [
            p for p in self.points3D.values()
            if p.track_length() >= min_track and p.error <= max_error
        ]

    # -------------------------------------------------
    # Camera centers (used for normalization)
    # -------------------------------------------------
    def camera_centers(self) -> np.ndarray:
        centers = [img.camera_center() for img in self.images.values()]
        return np.asarray(centers, dtype=np.float64)

    # -------------------------------------------------
    # Scene normalization (FOR NERF ONLY)
    # -------------------------------------------------
    def compute_normalization(self):
        centers = self.camera_centers()

        if len(centers) == 0:
            raise RuntimeError("No camera centers found")

        center = centers.mean(axis=0)

        dists = np.linalg.norm(centers - center, axis=1)

        # robust scale (avoid outliers)
        scale = 1.0 / np.median(dists)

        return center, scale


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
# LOADERS
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

            if model_id not in CAMERA_MODELS:
                raise RuntimeError(f"Unknown camera model id: {model_id}")

            model_name, n_params = CAMERA_MODELS[model_id]
            params = np.array(struct.unpack(f"<{n_params}d", f.read(8 * n_params)), dtype=np.float64)

            cameras[cam_id] = Camera(cam_id, model_name, width, height, params)

    return cameras


def _read_images_binary(path: Path) -> Dict[int, Image]:
    images = {}

    with open(path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]

        for _ in range(num):
            img_id = struct.unpack("<I", f.read(4))[0]

            qvec = np.array(struct.unpack("<4d", f.read(32)), dtype=np.float64)
            tvec = np.array(struct.unpack("<3d", f.read(24)), dtype=np.float64)

            cam_id = struct.unpack("<I", f.read(4))[0]

            # name
            name_bytes = bytearray()
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name_bytes.extend(c)
            name = name_bytes.decode("utf-8")

            num_points = struct.unpack("<Q", f.read(8))[0]

            xys = np.zeros((num_points, 2), dtype=np.float64)
            point_ids = np.full(num_points, -1, dtype=np.int64)

            for i in range(num_points):
                x, y = struct.unpack("<dd", f.read(16))
                pid = struct.unpack("<q", f.read(8))[0]

                xys[i] = [x, y]
                point_ids[i] = pid

            images[img_id] = Image(
                img_id, qvec, tvec, cam_id, name, xys, point_ids
            )

    return images


def _read_points3D_binary(path: Path) -> Dict[int, Point3D]:
    points = {}

    with open(path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]

        for _ in range(num):
            pid = struct.unpack("<Q", f.read(8))[0]

            xyz = np.array(struct.unpack("<3d", f.read(24)), dtype=np.float64)
            rgb = np.array(struct.unpack("<3B", f.read(3)), dtype=np.uint8)
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
        raise RuntimeError("No images in COLMAP model")

    for img in images.values():
        if img.camera_id not in cameras:
            raise RuntimeError(
                f"Image {img.id} references missing camera {img.camera_id}"
            )


# =====================================================
# MAIN
# =====================================================

def load_colmap_model(sparse_path: Path) -> ColmapModel:

    cams = sparse_path / "cameras.bin"
    imgs = sparse_path / "images.bin"
    pts = sparse_path / "points3D.bin"

    if not cams.exists():
        raise FileNotFoundError(f"Missing {cams}")
    if not imgs.exists():
        raise FileNotFoundError(f"Missing {imgs}")
    if not pts.exists():
        raise FileNotFoundError(f"Missing {pts}")

    cameras = _read_cameras_binary(cams)
    images = _read_images_binary(imgs)
    points = _read_points3D_binary(pts)

    _validate_model(cameras, images)

    return ColmapModel(cameras, images, points)