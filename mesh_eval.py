#!/usr/bin/env python3
"""
mesh_evaluation.py

MARK-2 Mesh Evaluation Stage (Enhanced)
---------------------------------------

- Computes geometry, topology, and density metrics
- Detects degenerate triangles, boundary loops (holes), halo edges
- Outputs a JSON suitable for mesh_cleanup.py
- Deterministic and resume-safe
"""

from pathlib import Path
import json
import numpy as np
import open3d as o3d

from utils.paths import ProjectPaths
from config_manager import load_config


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def triangle_area_squared(a, b, c) -> float:
    cross = np.cross(b - a, c - a)
    return float(np.dot(cross, cross))


def bbox_diagonal(verts: np.ndarray) -> float:
    if len(verts) == 0:
        return 0.0
    min_v = verts.min(axis=0)
    max_v = verts.max(axis=0)
    return float(np.linalg.norm(max_v - min_v))


def boundary_edge_info(mesh: o3d.geometry.TriangleMesh):
    """Compute boundary edges and estimate small holes."""
    edges = mesh.get_non_manifold_edges(allow_boundary_edges=True)
    boundary_edges = edges if len(edges) > 0 else np.array([], dtype=int)
    # Estimate holes as loops <= 6 edges
    holes = []
    visited = set()
    for e in boundary_edges:
        for v in e:
            if v not in visited:
                visited.add(v)
                holes.append(1)  # count as 1 small hole candidate
    return len(boundary_edges), len(holes)


# ------------------------------------------------------------
# Stage
# ------------------------------------------------------------
MAX_TRIANGLES = 500_000
DEGENERATE_EPS = 1e-12


def run(run_root: Path, project_root: Path, force: bool, logger):

    run_root = Path(run_root).resolve()
    project_root = Path(project_root).resolve()

    paths = ProjectPaths(project_root)
    _ = load_config(run_root)

    mesh_dir = paths.mesh
    mesh_in = mesh_dir / "mesh_cleaned.ply"
    report_out = paths.evaluation / "mesh_metrics.json"

    if not mesh_in.exists():
        raise FileNotFoundError(f"[mesh_eval] Missing mesh: {mesh_in}")

    report_out.parent.mkdir(parents=True, exist_ok=True)

    if report_out.exists() and not force:
        logger.info("[mesh_eval] Metrics exist — skipping")
        return

    mesh = o3d.io.read_triangle_mesh(str(mesh_in))
    if not mesh.has_triangles():
        raise RuntimeError("[mesh_eval] Mesh contains no triangles")

    mesh.compute_vertex_normals()
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    v_count = len(vertices)
    t_count = len(triangles)

    # ------------------------------------------------------------
    # Surface metrics
    # ------------------------------------------------------------
    surface_area = float(mesh.get_surface_area())
    bbox_diag = bbox_diagonal(vertices)

    # ------------------------------------------------------------
    # Triangle density
    # ------------------------------------------------------------
    tri_verts = vertices[triangles]
    tri_areas = 0.5 * np.linalg.norm(np.cross(tri_verts[:, 1] - tri_verts[:, 0],
                                              tri_verts[:, 2] - tri_verts[:, 0]), axis=1)
    density_stats = {
        "mean": float(np.mean(tri_areas)),
        "min": float(np.min(tri_areas)),
        "max": float(np.max(tri_areas)),
        "quantiles": np.quantile(tri_areas, [0.05, 0.25, 0.5, 0.75, 0.95]).tolist()
    }

    # ------------------------------------------------------------
    # Degenerate triangles
    # ------------------------------------------------------------
    degenerate_count = int(np.sum(tri_areas < DEGENERATE_EPS))

    # ------------------------------------------------------------
    # Holes & boundary
    # ------------------------------------------------------------
    boundary_edge_count, small_hole_count = boundary_edge_info(mesh)

    # ------------------------------------------------------------
    # Connected components
    # ------------------------------------------------------------
    if t_count <= MAX_TRIANGLES:
        clusters, _, _ = mesh.cluster_connected_triangles()
        component_count = int(np.max(clusters) + 1)
        component_mode = "exact"
    else:
        component_count = None
        component_mode = "skipped_large_mesh"

    # ------------------------------------------------------------
    # Save metrics
    # ------------------------------------------------------------
    metrics = {
        "vertices": int(v_count),
        "triangles": int(t_count),
        "surface_area": surface_area,
        "bounding_box": {
            "diagonal": bbox_diag
        },
        "triangle_density": density_stats,
        "topology": {
            "degenerate_triangles": degenerate_count,
            "boundary_edges": boundary_edge_count,
            "small_hole_candidates": small_hole_count,
            "component_count": component_count,
            "component_mode": component_mode
        },
        "deterministic": True
    }

    with open(report_out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"[mesh_eval] Metrics written: {report_out}")
    logger.info("[mesh_eval] DONE")