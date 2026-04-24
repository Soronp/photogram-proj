#!/usr/bin/env python3
"""
MARK-2 Evaluator (Robust v2)
---------------------------------
Improvements:
✔ Deterministic sampling
✔ Symmetric Chamfer Distance
✔ ICP fitness + RMSE
✔ Bounding-box scale normalization
✔ Clear metric naming
✔ Structured + explainable JSON output
✔ Separation of geometry + image proxies
"""

import open3d as o3d
import numpy as np
from pathlib import Path
import json
import sys
import struct
import subprocess

# ==============================
# CONFIG
# ==============================
THRESHOLD = 0.01
EPS = 1e-8
MAX_POINTS = 120000
SEED = 42

np.random.seed(SEED)

# ==============================
# INPUT
# ==============================
def ask_user():
    print("\n=== MARK-2 Evaluator (Robust v2) ===")

    ply_folder = Path(input("PLY folder: ").strip())
    sparse_model = Path(input("COLMAP sparse model: ").strip())
    output = Path(input("Output JSON: ").strip())

    if not ply_folder.exists() or not sparse_model.exists():
        print("[ERROR] Invalid paths")
        sys.exit(1)

    if output.suffix != ".json":
        output = output.with_suffix(".json")

    return ply_folder, sparse_model, output

# ==============================
# SAMPLING
# ==============================
def downsample(pts):
    if len(pts) > MAX_POINTS:
        idx = np.random.choice(len(pts), MAX_POINTS, replace=False)
        return pts[idx]
    return pts

# ==============================
# LOAD PLY
# ==============================
def load_ply(path):
    pcd = o3d.io.read_point_cloud(str(path))
    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        return None
    return downsample(pts)

# ==============================
# SCALE NORMALIZATION
# ==============================
def normalize_scale(pts):
    min_pt = pts.min(axis=0)
    max_pt = pts.max(axis=0)
    diag = np.linalg.norm(max_pt - min_pt) + EPS
    return pts / diag

# ==============================
# ALIGNMENT
# ==============================
def align_icp(src_pts, tgt_pts):
    src = o3d.geometry.PointCloud()
    tgt = o3d.geometry.PointCloud()

    src.points = o3d.utility.Vector3dVector(src_pts)
    tgt.points = o3d.utility.Vector3dVector(tgt_pts)

    # center
    src.translate(-src.get_center())
    tgt.translate(-tgt.get_center())

    reg = o3d.pipelines.registration.registration_icp(
        src, tgt,
        THRESHOLD,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )

    src.transform(reg.transformation)

    return (
        np.asarray(src.points),
        float(reg.fitness),
        float(reg.inlier_rmse)
    )

# ==============================
# NN DISTANCE
# ==============================
def nn_dist(a_pts, b_pts):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(b_pts)
    tree = o3d.geometry.KDTreeFlann(pcd)

    dists = []
    for p in a_pts:
        _, idx, _ = tree.search_knn_vector_3d(p, 1)
        if idx:
            nn = b_pts[idx[0]]
            dists.append(np.linalg.norm(p - nn))

    if not dists:
        return np.array([EPS])

    return np.array(dists)

# ==============================
# SYMMETRIC CHAMFER
# ==============================
def chamfer_symmetric(a, b):
    d1 = nn_dist(a, b)
    d2 = nn_dist(b, a)
    return {
        "mean": float(np.mean(d1) + np.mean(d2)),
        "std": float(np.std(d1) + np.std(d2))
    }

# ==============================
# COLMAP READER
# ==============================
def read_images_bin(path):
    cams = []
    with open(path, "rb") as f:
        num = struct.unpack("Q", f.read(8))[0]
        for _ in range(num):
            f.read(4)
            qw, qx, qy, qz = struct.unpack("dddd", f.read(32))
            tx, ty, tz = struct.unpack("ddd", f.read(24))
            f.read(4)

            while f.read(1) != b"\x00":
                pass

            n = struct.unpack("Q", f.read(8))[0]
            f.read(n * 24)

            cams.append((qw, qx, qy, qz, tx, ty, tz))

    return cams

# ==============================
# PROJECTION DISPERSION (CLEAR NAMING)
# ==============================
def projection_dispersion(points, cams):
    def project(p, cam):
        qw, qx, qy, qz, tx, ty, tz = cam

        R = np.array([
            [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
            [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
            [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy]
        ])
        t = np.array([tx, ty, tz])

        pc = (R @ p.T).T + t
        z = pc[:, 2] + EPS
        return np.stack([pc[:, 0] / z, pc[:, 1] / z], axis=1)

    spreads = []
    for cam in cams:
        proj = project(points, cam)
        center = np.mean(proj, axis=0)
        spreads.append(np.mean(np.linalg.norm(proj - center, axis=1)))

    return float(np.mean(spreads)) if spreads else float(EPS)

# ==============================
# RUN VISUALIZATION
# ==============================
def run_visualization(output_json):
    vis_script = Path(__file__).parent / "vis.py"

    if not vis_script.exists():
        print("[WARNING] vis.py not found. Skipping visualization.")
        return

    print("[INFO] Running visualization...")

    try:
        subprocess.run([
            sys.executable,
            str(vis_script),
            "--input",
            str(output_json),
            "--output_dir",
            str(output_json.parent / "viz_outputs")
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Visualization failed: {e}")


# ==============================
# MAIN
# ==============================
def main():
    ply_folder, sparse_model, output = ask_user()

    meshes = {}
    for f in ply_folder.glob("*.ply"):
        pts = load_ply(f)
        if pts is not None:
            pts = normalize_scale(pts)
            meshes[f.name] = pts

    if len(meshes) < 2:
        print("[ERROR] Need >= 2 meshes")
        sys.exit(1)

    cams = read_images_bin(sparse_model / "images.bin")

    results = {
        "pairwise": {},
        "per_model": {}
    }

    names = list(meshes.keys())

    # ------------------
    # Pairwise metrics
    # ------------------
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            print(f"Comparing {a} vs {b}")

            aligned_ab, fit_ab, rmse_ab = align_icp(meshes[a], meshes[b])
            aligned_ba, fit_ba, rmse_ba = align_icp(meshes[b], meshes[a])

            chamfer_ab = chamfer_symmetric(aligned_ab, meshes[b])
            chamfer_ba = chamfer_symmetric(aligned_ba, meshes[a])

            results["pairwise"][f"{a}__vs__{b}"] = {
                "chamfer_mean": (chamfer_ab["mean"] + chamfer_ba["mean"]) / 2,
                "chamfer_std": (chamfer_ab["std"] + chamfer_ba["std"]) / 2,
                "icp_fitness": (fit_ab + fit_ba) / 2,
                "icp_rmse": (rmse_ab + rmse_ba) / 2
            }

    # ------------------
    # Per-model metrics
    # ------------------
    for name, pts in meshes.items():
        print(f"Evaluating {name}")

        results["per_model"][name] = {
            "projection_dispersion": projection_dispersion(pts, cams),
            "num_points": int(len(pts))
        }

    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\n[INFO] Saved → {output}")

    # Auto visualization
    run_visualization(output)


if __name__ == "__main__":
    main()
