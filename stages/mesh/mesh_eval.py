#!/usr/bin/env python3

"""
Architectural Evaluation Stage

Evaluates reconstruction quality of generated mesh.

Metrics
-------
GF  Geometric Fidelity
SR  Structural Regularity
SI  Surface Integrity
DR  Detail Richness
SC  Spatial Coherence

Output
------
evaluation/architectural_metrics.json
"""

import json
import torch
import numpy as np
import open3d as o3d
from collections import defaultdict


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_SAMPLE_VERTS = 30000
K_NEIGHBORS = 32

WEIGHTS = {
    "GF": 0.2,
    "SR": 0.2,
    "SI": 0.2,
    "DR": 0.2,
    "SC": 0.2
}


# -----------------------------------------------------
# Utilities
# -----------------------------------------------------

def to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    return torch.tensor(x, dtype=torch.float32, device=DEVICE)


def bbox_diagonal(V):
    return torch.norm(V.max(dim=0).values - V.min(dim=0).values)


def sample_indices(N, max_n):

    if N <= max_n:
        return torch.arange(N, device=DEVICE)

    return torch.randperm(N, device=DEVICE)[:max_n]


def score_from_error(raw, scale):

    raw = to_tensor(raw)

    return float(torch.exp(-scale * raw))


def score_from_richness(raw, scale):

    raw = to_tensor(raw)

    return float(1.0 - torch.exp(-scale * raw))


# -----------------------------------------------------
# Geometric Fidelity
# -----------------------------------------------------

def compute_geometric_fidelity(V, F, bbox_diag):

    if F.shape[0] == 0:
        return 0.0

    v0 = V[F[:,0]]
    v1 = V[F[:,1]]
    v2 = V[F[:,2]]

    e0 = torch.norm(v0 - v1, dim=1)
    e1 = torch.norm(v1 - v2, dim=1)
    e2 = torch.norm(v2 - v0, dim=1)

    edges = torch.stack([e0,e1,e2], dim=1)

    edges /= (bbox_diag + 1e-12)

    edge_cv = edges.std() / (edges.mean() + 1e-12)

    longest = edges.max(dim=1).values
    shortest = edges.min(dim=1).values

    aspect = longest / (shortest + 1e-12)

    aspect_penalty = torch.mean(aspect - 1.0)

    error = edge_cv + 0.5 * aspect_penalty

    return score_from_error(error, scale=6)


# -----------------------------------------------------
# Structural Regularity
# -----------------------------------------------------

def compute_structural_regularity(V, bbox_diag):

    V_sample = V[sample_indices(V.shape[0], MAX_SAMPLE_VERTS)]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(V_sample.cpu().numpy())

    variances = []

    remaining = pcd

    for _ in range(4):

        if len(remaining.points) < 1000:
            break

        plane, inliers = remaining.segment_plane(
            distance_threshold = 0.002 * bbox_diag.item(),
            ransac_n = 3,
            num_iterations = 3000
        )

        if len(inliers) < 500:
            break

        pts = np.asarray(remaining.points)[inliers]

        variances.append(float(np.var(pts)))

        remaining = remaining.select_by_index(inliers, invert=True)

    if not variances:
        return 0.0, []

    mean_var = np.mean(variances)

    return score_from_error(mean_var, scale=800), variances


# -----------------------------------------------------
# Surface Integrity
# -----------------------------------------------------

def compute_surface_integrity(mesh):

    triangles = np.asarray(mesh.triangles)

    edge_count = defaultdict(int)

    for tri in triangles:

        edges = [
            tuple(sorted((tri[0], tri[1]))),
            tuple(sorted((tri[1], tri[2]))),
            tuple(sorted((tri[2], tri[0])))
        ]

        for e in edges:
            edge_count[e] += 1

    boundary_edges = sum(1 for e in edge_count if edge_count[e] == 1)

    non_manifold_edges = sum(1 for e in edge_count if edge_count[e] > 2)

    labels = np.array(mesh.cluster_connected_triangles()[0])

    components = labels.max() + 1 if labels.size else 1

    penalty = (
        0.001 * non_manifold_edges +
        0.001 * boundary_edges +
        0.2 * max(0, components - 1)
    )

    score = score_from_error(penalty, scale=10)

    return score, {
        "non_manifold_edges": int(non_manifold_edges),
        "boundary_edges": int(boundary_edges),
        "components": int(components)
    }


# -----------------------------------------------------
# Detail Richness
# -----------------------------------------------------

def compute_detail_richness(V, bbox_diag):

    V_sample = V[sample_indices(V.shape[0], MAX_SAMPLE_VERTS)]

    pts = V.cpu().numpy()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    kdtree = o3d.geometry.KDTreeFlann(pcd)

    curvatures = []

    for v in V_sample.cpu().numpy():

        _, idx, _ = kdtree.search_knn_vector_3d(v, K_NEIGHBORS)

        neighbors = pts[idx]

        centroid = neighbors.mean(axis=0)

        cov = np.cov((neighbors-centroid).T)

        eigvals = np.linalg.eigvalsh(cov)

        curvature = eigvals[0] / (eigvals.sum() + 1e-12)

        curvatures.append(float(curvature))

    curvatures = np.array(curvatures)

    mean_curvature = np.abs(curvatures).mean() / (bbox_diag.item()+1e-12)

    return score_from_richness(mean_curvature, scale=15), curvatures.tolist()


# -----------------------------------------------------
# Spatial Coherence
# -----------------------------------------------------

def compute_spatial_coherence(V, bbox_diag):

    pts = V.cpu().numpy()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    kdtree = o3d.geometry.KDTreeFlann(pcd)

    samples = pts[sample_indices(len(pts), MAX_SAMPLE_VERTS).cpu().numpy()]

    nn_dists = []

    for v in samples:

        _, _, d = kdtree.search_knn_vector_3d(v, 2)

        nn_dists.append(np.sqrt(d[1]))

    nn_dists = np.array(nn_dists)

    nn_dists /= (bbox_diag.item()+1e-12)

    cv = nn_dists.std() / (nn_dists.mean()+1e-12)

    return score_from_error(cv, scale=8), nn_dists.tolist()


# -----------------------------------------------------
# Mesh locator
# -----------------------------------------------------

def locate_mesh(paths):

    candidates = [
        paths.mesh / "mesh_cleaned.ply",
        paths.mesh / "mesh_raw.ply",
        paths.mesh / "mesh_hybrid.ply",
        paths.mesh / "mesh.ply",
        paths.mesh / "mesh.obj"
    ]

    for p in candidates:
        if p.exists():
            return p

    raise RuntimeError("No mesh found for evaluation")


# -----------------------------------------------------
# Stage
# -----------------------------------------------------

def run(paths, logger, tools, config):

    logger.info("[mesh_eval] starting")

    output_path = paths.evaluation / "architectural_metrics.json"

    if output_path.exists():
        logger.info("[mesh_eval] metrics exist — skipping")
        return

    mesh_path = locate_mesh(paths)

    logger.info(f"[mesh_eval] mesh: {mesh_path.name}")

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))

    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()

    V = torch.tensor(
        np.asarray(mesh.vertices),
        dtype=torch.float32,
        device=DEVICE
    )

    F = torch.tensor(
        np.asarray(mesh.triangles),
        dtype=torch.long,
        device=DEVICE
    )

    bbox_diag = bbox_diagonal(V)

    logger.info("[mesh_eval] computing metrics")

    GF = compute_geometric_fidelity(V,F,bbox_diag)

    SR, planes = compute_structural_regularity(V,bbox_diag)

    SI, topo = compute_surface_integrity(mesh)

    DR, curvature = compute_detail_richness(V,bbox_diag)

    SC, density = compute_spatial_coherence(V,bbox_diag)

    AVI = (
        WEIGHTS["GF"]*GF +
        WEIGHTS["SR"]*SR +
        WEIGHTS["SI"]*SI +
        WEIGHTS["DR"]*DR +
        WEIGHTS["SC"]*SC
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

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path,"w") as f:
        json.dump(metrics,f,indent=2)

    logger.info("[mesh_eval] completed")