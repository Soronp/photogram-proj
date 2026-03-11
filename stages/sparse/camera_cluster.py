#!/usr/bin/env python3
"""
camera_cluster.py

Stage 7 — Camera clustering

Responsibilities
----------------
• load selected sparse model
• compute camera centers
• cluster cameras spatially
• export cluster structure
"""

import json
import struct
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans


# --------------------------------------------------
# COLMAP binary helpers
# --------------------------------------------------

def read_next_bytes(fid, num_bytes, fmt, endian="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian + fmt, data)


def read_images_binary(path: Path):

    images = {}

    with open(path, "rb") as fid:

        num_reg_images = read_next_bytes(fid, 8, "Q")[0]

        for _ in range(num_reg_images):

            props = read_next_bytes(fid, 64, "idddddddi")

            image_id = props[0]
            qw, qx, qy, qz = props[1:5]
            tx, ty, tz = props[5:8]
            camera_id = props[8]

            name = ""

            while True:
                c = fid.read(1)
                if c == b"\x00":
                    break
                name += c.decode("utf-8")

            num_points = read_next_bytes(fid, 8, "Q")[0]
            fid.read(num_points * 24)

            images[image_id] = {
                "qvec": np.array([qw, qx, qy, qz]),
                "tvec": np.array([tx, ty, tz]),
                "camera_id": camera_id,
                "name": name
            }

    return images


# --------------------------------------------------
# Pose utilities
# --------------------------------------------------

def qvec2rotmat(qvec):

    w, x, y, z = qvec

    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y],
    ])


def camera_center(qvec, tvec):

    R = qvec2rotmat(qvec)

    return -R.T @ tvec


# --------------------------------------------------
# Sparse model loader
# --------------------------------------------------

def load_sparse_model(paths: Path):

    meta_file = paths.sparse / "export_ready.json"

    if not meta_file.exists():
        raise RuntimeError("Missing sparse metadata")

    meta = json.loads(meta_file.read_text())

    model_dir = paths.sparse / meta["model_dir"]

    if not model_dir.exists():
        raise RuntimeError("Sparse model directory missing")

    return model_dir


# --------------------------------------------------
# Camera clustering
# --------------------------------------------------

def cluster_cameras(centers, n_clusters):

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=0,
        n_init=10
    )

    labels = kmeans.fit_predict(centers)

    return labels


# --------------------------------------------------
# Stage execution
# --------------------------------------------------

def run(paths, logger, tools, config):

    logger.info("[camera_cluster] stage started")

    model_dir = load_sparse_model(paths)

    images_bin = model_dir / "images.bin"

    if not images_bin.exists():
        raise RuntimeError("images.bin missing in sparse model")

    logger.info(f"[camera_cluster] model: {model_dir.name}")

    images = read_images_binary(images_bin)

    logger.info(f"[camera_cluster] cameras detected: {len(images)}")

    centers = []
    ids = []

    for img_id, img in images.items():

        center = camera_center(img["qvec"], img["tvec"])

        centers.append(center)
        ids.append(img_id)

    centers = np.array(centers)

    n_images = len(centers)

    n_clusters = max(1, n_images // 200)

    logger.info(f"[camera_cluster] clusters: {n_clusters}")

    labels = cluster_cameras(centers, n_clusters)

    clusters = {}

    for img_id, label in zip(ids, labels):

        clusters.setdefault(str(label), []).append(int(img_id))

    output = {
        "num_clusters": int(n_clusters),
        "clusters": clusters
    }

    out_file = paths.sparse / "clusters.json"

    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"[camera_cluster] clusters saved: {out_file}")
    logger.info("[camera_cluster] stage completed")