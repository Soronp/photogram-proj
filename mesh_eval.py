#!/usr/bin/env python3
"""
MARK-2 GPU Architectural Evaluation Engine
Scientific Edition (GPU-Accelerated, Memory-Safe)

Outputs:
    evaluation/architectural_metrics.json
Compatible with MARK-2 visualization stage.
"""

from pathlib import Path
import json
import torch
import numpy as np
import open3d as o3d
from utils.paths import ProjectPaths


# ==========================================================
# GLOBAL SETTINGS
# ==========================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_SAMPLE_VERTS = 30_000
PLANE_ITER = 4
NEIGHBOR_SAMPLE = 64

WEIGHTS = {
    "GF": 0.2,
    "SR": 0.2,
    "SI": 0.2,
    "DR": 0.2,
    "SC": 0.2
}


# ==========================================================
# UTILITIES
# ==========================================================

def bbox_diagonal(V: torch.Tensor) -> torch.Tensor:
    return torch.norm(V.max(dim=0).values - V.min(dim=0).values)


def sample_indices(N, max_n):
    if N <= max_n:
        return torch.arange(N, device=DEVICE)
    return torch.randperm(N, device=DEVICE)[:max_n]


def score_from_error(raw, scale):
    return float(torch.exp(-scale * raw))


def score_from_richness(raw, scale):
    return float(1.0 - torch.exp(-scale * raw))


# ==========================================================
# 1️⃣ GEOMETRIC FIDELITY
# ==========================================================

def compute_geometric_fidelity(V, F, bbox_diag):
    if F.shape[0] == 0:
        return 0.0

    v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    edges = torch.cat([
        torch.norm(v0 - v1, dim=1),
        torch.norm(v1 - v2, dim=1),
        torch.norm(v2 - v0, dim=1)
    ])

    edges /= (bbox_diag + 1e-12)
    cv = edges.std() / (edges.mean() + 1e-12)

    return score_from_error(cv, scale=6)


# ==========================================================
# 2️⃣ STRUCTURAL REGULARITY
# ==========================================================

def compute_structural_regularity(V, bbox_diag):
    V_sample = V[sample_indices(V.shape[0], MAX_SAMPLE_VERTS)]
    remaining = V_sample.clone()

    variances = []
    threshold = 0.002 * bbox_diag

    for _ in range(PLANE_ITER):

        if remaining.shape[0] < 1000:
            break

        N_trials = min(5000, remaining.shape[0])
        idx = torch.randint(0, remaining.shape[0], (N_trials, 3), device=DEVICE)
        pts = remaining[idx]

        p1, p2, p3 = pts[:, 0], pts[:, 1], pts[:, 2]

        normal = torch.cross(p2 - p1, p3 - p1, dim=1)
        norm_len = torch.norm(normal, dim=1, keepdim=True)

        valid = norm_len.squeeze() > 1e-6
        if valid.sum() == 0:
            break

        normal = normal[valid] / norm_len[valid]
        p1 = p1[valid]
        d = -(normal * p1).sum(dim=1)

        distances = torch.abs(remaining @ normal.T + d)
        inlier_counts = (distances < threshold).sum(dim=0)

        best_idx = inlier_counts.argmax().item()
        best_distances = distances[:, best_idx]

        inliers = best_distances < threshold
        if inliers.sum() < 500:
            break

        variances.append(best_distances[inliers].var().item())
        remaining = remaining[~inliers]

    if not variances:
        return 0.0, []

    mean_var = torch.tensor(np.mean(variances), device=DEVICE)
    return score_from_error(mean_var, scale=800), variances


# ==========================================================
# 3️⃣ SURFACE INTEGRITY
# ==========================================================

def compute_surface_integrity(F):
    if F.shape[0] == 0:
        return 0.0, {
            "non_manifold_edges": 0,
            "boundary_edges": 0,
            "components": 1
        }

    E = torch.cat([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], dim=0)
    E = torch.sort(E, dim=1).values
    _, counts = torch.unique(E, dim=0, return_counts=True)

    boundary_edges = (counts == 1).sum().item()
    non_manifold_edges = (counts > 2).sum().item()
    components = 1

    penalty = (
        0.001 * non_manifold_edges +
        0.001 * boundary_edges +
        0.2 * max(0, components - 1)
    )

    return score_from_error(torch.tensor(penalty, device=DEVICE), scale=10), {
        "non_manifold_edges": non_manifold_edges,
        "boundary_edges": boundary_edges,
        "components": components
    }


# ==========================================================
# 4️⃣ DETAIL RICHNESS
# ==========================================================

def compute_detail_richness(V, bbox_diag):
    V_sample = V[sample_indices(V.shape[0], MAX_SAMPLE_VERTS)]

    H_vals = []

    for i in range(V_sample.shape[0]):
        idx = torch.randint(0, V.shape[0], (NEIGHBOR_SAMPLE,), device=DEVICE)
        pts = V[idx]

        centroid = pts.mean(dim=0)
        cov = (pts - centroid).T @ (pts - centroid) / NEIGHBOR_SAMPLE

        eigvals = torch.linalg.eigvalsh(cov)
        curvature = eigvals[0] / (eigvals.sum() + 1e-12)

        H_vals.append(curvature)

    H_vals = torch.stack(H_vals) / (bbox_diag + 1e-12)
    mean_curvature = H_vals.abs().mean()

    return (
        score_from_richness(mean_curvature, scale=15),
        H_vals.cpu().tolist()
    )


# ==========================================================
# 5️⃣ SPATIAL COHERENCE
# ==========================================================

def compute_spatial_coherence(V, bbox_diag):
    V_sample = V[sample_indices(V.shape[0], MAX_SAMPLE_VERTS)]

    mean_dist_list = []

    for i in range(V_sample.shape[0]):
        idx = torch.randint(0, V.shape[0], (NEIGHBOR_SAMPLE,), device=DEVICE)
        pts = V[idx]

        dists = torch.norm(V_sample[i] - pts, dim=1)
        mean_dist_list.append(dists.mean())

    mean_dist = torch.tensor(mean_dist_list, device=DEVICE)
    mean_dist /= (bbox_diag + 1e-12)

    cv = mean_dist.std() / (mean_dist.mean() + 1e-12)

    return score_from_error(cv, scale=8), mean_dist.cpu().tolist()


# ==========================================================
# MAIN
# ==========================================================

def run(run_root: Path, project_root: Path, force: bool, logger):

    paths = ProjectPaths(project_root)

    mesh_path = paths.mesh / "mesh_cleaned.ply"
    output_path = paths.evaluation / "architectural_metrics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()

    V = torch.tensor(np.asarray(mesh.vertices), device=DEVICE, dtype=torch.float32)
    F = torch.tensor(np.asarray(mesh.triangles), device=DEVICE, dtype=torch.long)

    bbox_diag = bbox_diagonal(V)

    logger.info("MARK-2 GPU Scientific Evaluation Started")

    GF = compute_geometric_fidelity(V, F, bbox_diag)
    SR, planes = compute_structural_regularity(V, bbox_diag)
    SI, topo = compute_surface_integrity(F)
    DR, curvature = compute_detail_richness(V, bbox_diag)
    SC, density = compute_spatial_coherence(V, bbox_diag)

    AVI = (
        WEIGHTS["GF"] * GF +
        WEIGHTS["SR"] * SR +
        WEIGHTS["SI"] * SI +
        WEIGHTS["DR"] * DR +
        WEIGHTS["SC"] * SC
    )

    metrics = {
        "architectural_metrics": {
            "geometric_fidelity": GF,
            "structural_regularity": SR,
            "surface_integrity": SI,
            "detail_richness": DR,
            "spatial_coherence": SC,
            "architectural_value_index": AVI
        },
        "diagnostics": {
            "plane_variances": planes,
            "curvature": curvature,
            "density": density,
            "topology": topo
        }
    }

    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("MARK-2 GPU Scientific Evaluation Complete")