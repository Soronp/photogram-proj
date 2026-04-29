#!/usr/bin/env python3
"""
MARK-2 Evaluator (v12 - Dual Mode: GT + GT-Free)
------------------------------------------------
✔ GT-based evaluation (true geometry)
✔ GT-free relative evaluation (consensus-based)
✔ Identical metric schema across modes
✔ Stable F-score (scale-aware)
✔ Robust coverage (distribution-based)
✔ Paper-aligned outputs
"""

import open3d as o3d
import numpy as np
from pathlib import Path
import json
import sys

EPS = 1e-8
MAX_POINTS = 120000
np.random.seed(42)


# ==============================
# INPUT
# ==============================
def ask_user():
    print("\n=== MARK-2 Evaluator (v12 Dual Mode) ===")

    ply_folder = Path(input("PLY folder: ").strip())
    gt_input = input("Ground truth PLY (leave empty for GT-free): ").strip()
    output = Path(input("Output JSON file: ").strip())

    if not ply_folder.exists():
        print("[ERROR] Invalid PLY folder")
        sys.exit(1)

    gt_path = Path(gt_input) if gt_input else None

    if gt_path and not gt_path.exists():
        print("[ERROR] Invalid GT path")
        sys.exit(1)

    if output.suffix != ".json":
        output = output.with_suffix(".json")

    return ply_folder, gt_path, output.resolve()


# ==============================
# UTIL
# ==============================
def downsample(pts):
    if len(pts) > MAX_POINTS:
        idx = np.random.choice(len(pts), MAX_POINTS, replace=False)
        return pts[idx]
    return pts


def load_ply(path):
    pcd = o3d.io.read_point_cloud(str(path))
    pts = np.asarray(pcd.points)

    if len(pts) == 0:
        print(f"[WARN] Empty → {path.name}")
        return None

    return downsample(pts)


def scene_scale(pts):
    return np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))


# ==============================
# ICP ALIGNMENT
# ==============================
def align_icp(src, tgt, threshold):
    s = o3d.geometry.PointCloud()
    t = o3d.geometry.PointCloud()

    s.points = o3d.utility.Vector3dVector(src)
    t.points = o3d.utility.Vector3dVector(tgt)

    s.translate(-s.get_center())
    t.translate(-t.get_center())

    reg = o3d.pipelines.registration.registration_icp(
        s, t,
        threshold,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )

    s.transform(reg.transformation)

    return np.asarray(s.points), float(reg.fitness), float(reg.inlier_rmse)


# ==============================
# DISTANCE CORE
# ==============================
def nn_dist(a, b):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(b)

    tree = o3d.geometry.KDTreeFlann(pcd)

    dists = []
    for p in a:
        _, idx, _ = tree.search_knn_vector_3d(p, 1)
        if idx:
            dists.append(np.linalg.norm(p - b[idx[0]]))

    return np.array(dists) if len(dists) else np.array([EPS])


# ==============================
# COVERAGE (ROBUST)
# ==============================
def coverage_ratio(comp_dist):
    if len(comp_dist) == 0:
        return 0.0

    thr = np.percentile(comp_dist, 90)
    return float(np.mean(comp_dist <= thr))


# ==============================
# DISTRIBUTION
# ==============================
def error_distribution(acc, comp):
    all_err = np.concatenate([acc, comp])

    return {
        "acc_std": float(np.std(acc)),
        "comp_std": float(np.std(comp)),
        "error_p50": float(np.percentile(all_err, 50)),
        "error_p90": float(np.percentile(all_err, 90)),
        "error_p95": float(np.percentile(all_err, 95)),
        "error_max": float(np.max(all_err))
    }


# ==============================
# STABLE F-SCORE
# ==============================
def stable_fscore(acc, comp, scale):
    acc_n = acc / (scale + EPS)
    comp_n = comp / (scale + EPS)

    precision = np.mean(1.0 / (1.0 + acc_n))
    recall = np.mean(1.0 / (1.0 + comp_n))

    return float(2 * precision * recall / (precision + recall + EPS))


# ==============================
# METRIC CORE
# ==============================
def compute_metrics(pred, ref):
    acc = nn_dist(pred, ref)
    comp = nn_dist(ref, pred)

    scale = scene_scale(ref)

    return {
        "accuracy_mean": float(np.mean(acc)),
        "completeness_mean": float(np.mean(comp)),
        "chamfer_distance": float(np.mean(acc) + np.mean(comp)),
        "coverage_ratio": coverage_ratio(comp),
        "fscore": stable_fscore(acc, comp, scale),
        **error_distribution(acc, comp)
    }


# ==============================
# PSEUDO GT (CONSENSUS)
# ==============================
def build_consensus(models):
    print("[INFO] Building pseudo-GT (consensus)...")

    all_points = np.vstack(list(models.values()))

    # downsample to keep stable
    return downsample(all_points)


# ==============================
# MAIN
# ==============================
def main():
    ply_folder, gt_path, output = ask_user()

    meshes = {}

    print("\n[INFO] Loading PLY files...")

    for f in ply_folder.glob("*.ply"):
        if gt_path and f.resolve() == gt_path.resolve():
            print(f"[INFO] Skipping GT → {f.name}")
            continue

        pts = load_ply(f)
        if pts is not None:
            meshes[f.name] = pts

    if not meshes:
        print("[ERROR] No models found")
        sys.exit(1)

    # ==========================
    # MODE SELECTION
    # ==========================
    if gt_path:
        mode = "GT"
        ref = load_ply(gt_path)
    else:
        mode = "RELATIVE"
        ref = build_consensus(meshes)

    scale = scene_scale(ref)
    threshold = 0.02 * scale

    results = {
        "evaluation_protocol": {
            "mode": mode,
            "scene_scale": float(scale),
            "icp_threshold": float(threshold)
        },
        "per_model_metrics": {}
    }

    print(f"\n[INFO] Mode: {mode}")
    print("[INFO] Evaluating...")

    for name, pts in meshes.items():
        aligned, fit, rmse = align_icp(pts, ref, threshold)

        m = compute_metrics(aligned, ref)

        results["per_model_metrics"][name] = {
            **m,
            "icp_fitness": fit,
            "icp_rmse": rmse,
            "num_points": len(pts)
        }

    with open(output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(f"\n[INFO] Saved → {output}")


if __name__ == "__main__":
    main()