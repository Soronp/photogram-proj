#!/usr/bin/env python3
"""
visualization.py

MARK-2 Visualization Stage
--------------------------
- Produces human-readable visual artifacts
- Does NOT modify any upstream data
- Deterministic and restart-safe
- Outputs PNG renders only
- Pipeline-compatible: run_visualization(project_root, force)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from utils.logger import get_logger
from utils.paths import ProjectPaths
from utils.config import load_config


# ------------------------------------------------------------------
# Visualization
# ------------------------------------------------------------------

def run_visualization(project_root: Path, force: bool = False):
    paths = ProjectPaths(project_root)
    _ = load_config(project_root)
    logger = get_logger("visualization", project_root)

    mesh_path = paths.mesh / "mesh_cleaned.ply"
    eval_summary = paths.evaluation / "summary.json"
    vis_dir = paths.visualization

    logger.info("=== Visualization Stage ===")
    logger.info(f"Visualization dir: {vis_dir}")

    vis_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # Load mesh (if available)
    # --------------------------------------------------

    if not mesh_path.exists():
        logger.warning("Cleaned mesh not found — skipping mesh visualizations")
        return

    logger.info("Loading cleaned mesh")
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    mesh.compute_vertex_normals()

    # --------------------------------------------------
    # Mesh render
    # --------------------------------------------------

    logger.info("Rendering mesh overview")
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(mesh)
    vis.poll_events()
    vis.update_renderer()

    img_path = vis_dir / "mesh_overview.png"
    vis.capture_screen_image(str(img_path))
    vis.destroy_window()
    logger.info(f"Saved mesh overview: {img_path}")

    # --------------------------------------------------
    # Bounding box render
    # --------------------------------------------------

    logger.info("Rendering bounding box")
    bbox = mesh.get_axis_aligned_bounding_box()
    bbox.color = (1, 0, 0)

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(mesh)
    vis.add_geometry(bbox)
    vis.poll_events()
    vis.update_renderer()

    bbox_img = vis_dir / "mesh_bbox.png"
    vis.capture_screen_image(str(bbox_img))
    vis.destroy_window()
    logger.info(f"Saved bounding box render: {bbox_img}")

    # --------------------------------------------------
    # Triangle density visualization
    # --------------------------------------------------

    logger.info("Generating triangle density visualization")
    tris = np.asarray(mesh.triangles)
    verts = np.asarray(mesh.vertices)
    tri_centers = verts[tris].mean(axis=1)
    z_vals = tri_centers[:, 2]

    plt.figure(figsize=(6, 4))
    plt.hist(z_vals, bins=100)
    plt.title("Triangle Z-Density Distribution")
    plt.xlabel("Z")
    plt.ylabel("Triangle Count")

    density_img = vis_dir / "triangle_density.png"
    plt.tight_layout()
    plt.savefig(density_img, dpi=200)
    plt.close()
    logger.info(f"Saved triangle density plot: {density_img}")

    # --------------------------------------------------
    # Metrics text panel
    # --------------------------------------------------

    if eval_summary.exists():
        logger.info("Rendering metrics summary panel")
        with open(eval_summary, "r", encoding="utf-8") as f:
            summary = json.load(f)

        text_lines = []
        for stage, data in summary.get("stages", {}).items():
            text_lines.append(f"[{stage.upper()}]")
            for k, v in data.items():
                if isinstance(v, (int, float, bool)):
                    text_lines.append(f"{k}: {v}")
            text_lines.append("")

        plt.figure(figsize=(8, 10))
        plt.axis("off")
        plt.text(0.01, 0.99, "\n".join(text_lines),
                 va="top", ha="left", family="monospace")

        metrics_img = vis_dir / "metrics_summary.png"
        plt.savefig(metrics_img, dpi=200, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved metrics summary panel: {metrics_img}")
    else:
        logger.warning("Evaluation summary not found — skipping metrics panel")

    logger.info("Visualization stage completed successfully")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MARK-2 Visualization")
    parser.add_argument("project_root", type=Path)
    parser.add_argument("--force", action="store_true", help="Force re-rendering")
    args = parser.parse_args()

    run_visualization(args.project_root, args.force)


if __name__ == "__main__":
    main()
