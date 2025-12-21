#!/usr/bin/env python3
"""
visualization.py

MARK-2 Visualization Stage
--------------------------
- Produces human-readable visual artifacts
- Does NOT modify any upstream data
- Deterministic and restart-safe
- Outputs PNG renders only
"""

import json
from pathlib import Path

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from utils.paths import ProjectPaths
from utils.config import load_config


# --------------------------------------------------
# PIPELINE STAGE
# --------------------------------------------------
def run(run_root: Path, project_root: Path, force: bool, logger):
    paths = ProjectPaths(project_root)
    load_config(project_root)

    vis_dir = paths.visualization
    mesh_path = paths.mesh / "mesh_cleaned.ply"
    eval_summary = paths.evaluation / "summary.json"

    logger.info("[vis] Visualization stage started")
    vis_dir.mkdir(parents=True, exist_ok=True)

    if not mesh_path.exists():
        logger.info("[vis] No cleaned mesh found — skipping")
        return

    logger.info("[vis] Loading mesh")
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    mesh.compute_vertex_normals()

    # --------------------------------------------------
    # Mesh overview
    # --------------------------------------------------
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(mesh)
    vis.poll_events()
    vis.update_renderer()

    out = vis_dir / "mesh_overview.png"
    vis.capture_screen_image(str(out))
    vis.destroy_window()
    logger.info(f"[vis] Saved {out.name}")

    # --------------------------------------------------
    # Bounding box
    # --------------------------------------------------
    bbox = mesh.get_axis_aligned_bounding_box()
    bbox.color = (1, 0, 0)

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(mesh)
    vis.add_geometry(bbox)
    vis.poll_events()
    vis.update_renderer()

    out = vis_dir / "mesh_bbox.png"
    vis.capture_screen_image(str(out))
    vis.destroy_window()
    logger.info(f"[vis] Saved {out.name}")

    # --------------------------------------------------
    # Triangle Z-density
    # --------------------------------------------------
    tris = np.asarray(mesh.triangles)
    verts = np.asarray(mesh.vertices)
    z_vals = verts[tris].mean(axis=1)[:, 2]

    plt.figure(figsize=(6, 4))
    plt.hist(z_vals, bins=100)
    plt.title("Triangle Z-Density")
    plt.xlabel("Z")
    plt.ylabel("Count")

    out = vis_dir / "triangle_density.png"
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    logger.info(f"[vis] Saved {out.name}")

    # --------------------------------------------------
    # Metrics summary
    # --------------------------------------------------
    if eval_summary.exists():
        with open(eval_summary, "r", encoding="utf-8") as f:
            summary = json.load(f)

        lines = []
        for stage, data in summary.get("stages", {}).items():
            lines.append(f"[{stage.upper()}]")
            for k, v in data.items():
                if isinstance(v, (int, float, bool)):
                    lines.append(f"{k}: {v}")
            lines.append("")

        plt.figure(figsize=(8, 10))
        plt.axis("off")
        plt.text(0.01, 0.99, "\n".join(lines),
                 va="top", ha="left", family="monospace")

        out = vis_dir / "metrics_summary.png"
        plt.savefig(out, dpi=200, bbox_inches="tight")
        plt.close()
        logger.info(f"[vis] Saved {out.name}")

    logger.info("[vis] Visualization completed")
