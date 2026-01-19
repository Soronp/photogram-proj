#!/usr/bin/env python3
"""
mesh_cleanup.py

MARK-2 Mesh Cleanup Stage (Authoritative, Mature)
------------------------------------------------
- Deterministic removal of stray and deformed mesh clusters
- Multi-signal cluster filtering (area, density, compactness, distance)
- Config-driven, run-scoped
- No ML, no heuristics that break determinism
"""

from pathlib import Path
import json
from typing import Dict, Tuple

import numpy as np
import open3d as o3d

from utils.paths import ProjectPaths
from config_manager import load_config


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def triangle_area(a, b, c) -> float:
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a))


def bbox_volume(verts: np.ndarray) -> float:
    if len(verts) == 0:
        return 0.0
    min_v = verts.min(axis=0)
    max_v = verts.max(axis=0)
    return float(np.prod(max_v - min_v))


def centroid(verts: np.ndarray) -> np.ndarray:
    return verts.mean(axis=0) if len(verts) else np.zeros(3)


# ------------------------------------------------------------
# Main stage
# ------------------------------------------------------------

def run(run_root: Path, project_root: Path, force: bool, logger):
    run_root = run_root.resolve()
    project_root = project_root.resolve()

    paths = ProjectPaths(project_root)
    logger.info("[mesh_cleanup] START")

    config = load_config(run_root, logger)
    mesh_cfg = config.get("mesh", {})

    mesh_dir = paths.mesh
    mesh_in = mesh_dir / "mesh_raw.ply"
    mesh_out = mesh_dir / "mesh_cleaned.ply"
    report_out = mesh_dir / "mesh_cleanup_report.json"

    if not mesh_in.exists():
        raise FileNotFoundError(f"Missing mesh: {mesh_in}")

    if mesh_out.exists() and not force:
        logger.info("[mesh_cleanup] Output exists — skipping")
        return

    mesh = o3d.io.read_triangle_mesh(str(mesh_in))
    if not mesh.has_triangles():
        raise RuntimeError("Mesh contains no triangles")

    mesh.compute_vertex_normals()

    # ------------------------------------------------------------
    # Topological cleanup (safe, deterministic)
    # ------------------------------------------------------------

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    # ------------------------------------------------------------
    # Cluster analysis
    # ------------------------------------------------------------

    tri_clusters, _, _ = mesh.cluster_connected_triangles()
    tri_clusters = np.asarray(tri_clusters)

    cluster_stats: Dict[int, Dict] = {}

    for i, cid in enumerate(tri_clusters):
        tri = triangles[i]
        verts = vertices[tri]

        stat = cluster_stats.setdefault(
            cid,
            {
                "area": 0.0,
                "triangles": [],
                "vertex_indices": set(),
            },
        )

        stat["area"] += triangle_area(*verts)
        stat["triangles"].append(i)
        stat["vertex_indices"].update(tri)

    # Finalize cluster metrics
    for cid, stat in cluster_stats.items():
        v_idx = np.array(list(stat["vertex_indices"]))
        v = vertices[v_idx]

        stat["vertex_count"] = len(v_idx)
        stat["bbox_volume"] = bbox_volume(v)
        stat["centroid"] = centroid(v)
        stat["density"] = (
            stat["area"] / stat["bbox_volume"]
            if stat["bbox_volume"] > 0
            else 0.0
        )

    # ------------------------------------------------------------
    # Identify dominant (target) cluster
    # ------------------------------------------------------------

    main_cluster = max(cluster_stats, key=lambda c: cluster_stats[c]["area"])
    main = cluster_stats[main_cluster]

    main_area = main["area"]
    main_vertices = main["vertex_count"]
    main_bbox = main["bbox_volume"]
    main_centroid = main["centroid"]

    # ------------------------------------------------------------
    # Filtering thresholds (configurable, safe defaults)
    # ------------------------------------------------------------

    min_area_ratio = float(mesh_cfg.get("min_component_area_ratio", 0.02))
    min_vertex_ratio = float(mesh_cfg.get("min_component_vertex_ratio", 0.05))
    min_bbox_ratio = float(mesh_cfg.get("min_component_bbox_ratio", 0.02))
    max_centroid_dist_ratio = float(mesh_cfg.get("max_component_distance_ratio", 2.5))

    remove_triangles = []

    for cid, stat in cluster_stats.items():
        if cid == main_cluster:
            continue

        area_bad = stat["area"] < main_area * min_area_ratio
        vertex_bad = stat["vertex_count"] < main_vertices * min_vertex_ratio
        bbox_bad = stat["bbox_volume"] < main_bbox * min_bbox_ratio

        dist = np.linalg.norm(stat["centroid"] - main_centroid)
        main_extent = np.cbrt(main_bbox) if main_bbox > 0 else 1.0
        distance_bad = dist > main_extent * max_centroid_dist_ratio

        # CRITICAL: remove if ANY two conditions agree
        bad_votes = sum([area_bad, vertex_bad, bbox_bad, distance_bad])

        if bad_votes >= 2:
            remove_triangles.extend(stat["triangles"])

    # ------------------------------------------------------------
    # Apply removal
    # ------------------------------------------------------------

    if remove_triangles:
        mesh.remove_triangles_by_index(remove_triangles)
        mesh.remove_unreferenced_vertices()

    # ------------------------------------------------------------
    # Optional smoothing
    # ------------------------------------------------------------

    if mesh_cfg.get("smoothing", False):
        iters = int(mesh_cfg.get("smoothing_iterations", 5))
        mesh = mesh.filter_smooth_laplacian(iters)
        mesh.compute_vertex_normals()

    # ------------------------------------------------------------
    # Optional decimation
    # ------------------------------------------------------------

    ratio = mesh_cfg.get("decimation_ratio")
    if ratio is not None and 0.0 < ratio < 1.0:
        target = int(len(mesh.triangles) * ratio)
        mesh = mesh.simplify_quadric_decimation(target)
        mesh.compute_vertex_normals()

    # ------------------------------------------------------------
    # Persist outputs
    # ------------------------------------------------------------

    o3d.io.write_triangle_mesh(str(mesh_out), mesh, write_ascii=False)

    with open(report_out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "input_triangles": len(triangles),
                "output_triangles": len(mesh.triangles),
                "clusters_detected": len(cluster_stats),
                "dominant_cluster": int(main_cluster),
                "filters": {
                    "min_area_ratio": min_area_ratio,
                    "min_vertex_ratio": min_vertex_ratio,
                    "min_bbox_ratio": min_bbox_ratio,
                    "max_distance_ratio": max_centroid_dist_ratio,
                },
            },
            f,
            indent=2,
        )

    logger.info("[mesh_cleanup] DONE")
