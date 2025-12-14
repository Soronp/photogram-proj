#!/usr/bin/env python3
"""
mesh_evaluation.py

MARK-2 Mesh Evaluation Stage (SAFE)
----------------------------------
- Computes scalable mesh health metrics
- NO unbounded topology operations
- Guaranteed termination on large meshes
- Deterministic and restart-safe
- Input : mesh/mesh_cleaned.ply
- Output: evaluation/mesh_metrics.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d

from utils.logger import get_logger
from utils.paths import ProjectPaths
from utils.config import load_config


# ------------------------------------------------------------------
# Configuration (hard safety limits)
# ------------------------------------------------------------------

MAX_TOPOLOGY_TRIANGLES = 300_000
DEGENERATE_EPS = 1e-12


# ------------------------------------------------------------------
# Mesh evaluation
# ------------------------------------------------------------------

def run_mesh_evaluation(project_root: Path):
    paths = ProjectPaths(project_root)
    _ = load_config(project_root)
    logger = get_logger("mesh_evaluation", project_root)

    mesh_in = paths.mesh / "mesh_cleaned.ply"
    eval_dir = paths.evaluation
    metrics_out = eval_dir / "mesh_metrics.json"

    logger.info("=== Mesh Evaluation Stage ===")
    logger.info(f"Mesh input : {mesh_in}")
    logger.info(f"Metrics out: {metrics_out}")

    if not mesh_in.exists():
        raise FileNotFoundError(f"Cleaned mesh not found: {mesh_in}")

    eval_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # Load mesh
    # --------------------------------------------------

    logger.info("Loading cleaned mesh")
    mesh = o3d.io.read_triangle_mesh(str(mesh_in))

    if not mesh.has_triangles():
        raise RuntimeError("Mesh contains no triangles")

    mesh.compute_vertex_normals()

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    v_count = len(vertices)
    t_count = len(triangles)

    logger.info(
        f"Loaded mesh: {v_count:,} vertices, {t_count:,} triangles"
    )

    # --------------------------------------------------
    # Basic geometric metrics (SAFE)
    # --------------------------------------------------

    surface_area = float(mesh.get_surface_area())

    aabb = mesh.get_axis_aligned_bounding_box()
    bbox_extent = aabb.get_extent()
    bbox_diagonal = float(np.linalg.norm(bbox_extent))

    # --------------------------------------------------
    # Degenerate triangle estimation (VECTORISED)
    # --------------------------------------------------

    logger.info("Estimating degenerate triangles (vectorized)")

    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]

    cross = np.cross(v1 - v0, v2 - v0)
    area2 = np.einsum("ij,ij->i", cross, cross)

    degenerate_count = int(np.sum(area2 < DEGENERATE_EPS))

    # --------------------------------------------------
    # Connected component estimation (SAFE MODE)
    # --------------------------------------------------

    logger.info("Estimating connected components")

    if t_count <= MAX_TOPOLOGY_TRIANGLES:
        clusters, _, _ = mesh.cluster_connected_triangles()
        component_count = int(np.max(clusters) + 1)
        component_mode = "exact"
    else:
        component_count = None
        component_mode = "skipped_large_mesh"
        logger.warning(
            f"Connected component analysis skipped "
            f"(triangles={t_count:,} > {MAX_TOPOLOGY_TRIANGLES:,})"
        )

    # --------------------------------------------------
    # Topology flags (SAFE HEURISTICS)
    # --------------------------------------------------

    topology = {
        "degenerate_triangles": degenerate_count,
        "component_count": component_count,
        "component_mode": component_mode,
        "edge_manifold": None,
        "vertex_manifold": None,
        "self_intersecting": None,
        "notes": "Expensive global topology checks skipped for scalability",
    }

    # --------------------------------------------------
    # Metrics report
    # --------------------------------------------------

    metrics = {
        "vertices": int(v_count),
        "triangles": int(t_count),
        "surface_area": surface_area,
        "bounding_box": {
            "extent": bbox_extent.tolist(),
            "diagonal": bbox_diagonal,
        },
        "topology": topology,
        "deterministic": True,
        "safe_mode": True,
    }

    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Mesh evaluation completed successfully")
    logger.info(f"Metrics written to: {metrics_out}")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MARK-2 Mesh Evaluation (Safe)")
    parser.add_argument("project_root", type=Path)

    args = parser.parse_args()
    run_mesh_evaluation(args.project_root)


if __name__ == "__main__":
    main()
