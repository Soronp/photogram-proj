#!/usr/bin/env python3
"""
mesh_evaluation.py

MARK-2 Mesh Evaluation Stage (Canonical)
----------------------------------------

- Uses ProjectPaths (single filesystem authority)
- Deterministic
- Resume-safe
- Safe for large meshes
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
    """Squared area of a triangle (for degenerate detection)."""
    cross = np.cross(b - a, c - a)
    return float(np.dot(cross, cross))


def bbox_diagonal(verts: np.ndarray) -> float:
    """Compute diagonal length of axis-aligned bounding box."""
    if len(verts) == 0:
        return 0.0
    min_v = verts.min(axis=0)
    max_v = verts.max(axis=0)
    return float(np.linalg.norm(max_v - min_v))


# ------------------------------------------------------------
# Stage
# ------------------------------------------------------------

MAX_TRIANGLES = 300_000
DEGENERATE_EPS = 1e-12


def run(run_root: Path, project_root: Path, force: bool, logger):

    run_root = Path(run_root).resolve()
    project_root = Path(project_root).resolve()

    paths = ProjectPaths(project_root)
    config = load_config(run_root, logger)
    mesh_cfg = config.get("mesh", {})

    logger.info("[mesh_eval] START")

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
    # Geometry metrics
    # ------------------------------------------------------------

    surface_area = float(mesh.get_surface_area())
    bbox_diag = bbox_diagonal(vertices)

    # ------------------------------------------------------------
    # Degenerate triangles
    # ------------------------------------------------------------

    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]

    area2 = np.einsum("ij,ij->i", np.cross(v1 - v0, v2 - v0), np.cross(v1 - v0, v2 - v0))
    degenerate_count = int(np.sum(area2 < DEGENERATE_EPS))

    # ------------------------------------------------------------
    # Connected components (safe for large meshes)
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
        "topology": {
            "degenerate_triangles": degenerate_count,
            "component_count": component_count,
            "component_mode": component_mode
        },
        "deterministic": True
    }

    with open(report_out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info("[mesh_eval] DONE")