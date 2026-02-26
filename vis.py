#!/usr/bin/env python3
"""
MARK-2 Visualization Stage (Robust + Version Safe)

Enhancements:
- Unified legend box system with safe wrapping
- Dynamic legend sizing (no overflow possible)
- Clean aligned layout
- Fully descriptive spider chart legend
- No convention changes
"""

from pathlib import Path
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import textwrap

from utils.paths import ProjectPaths
from utils.config import load_config


# ==========================================================
# UTILITIES
# ==========================================================

def clone_mesh(mesh):
    return o3d.geometry.TriangleMesh(mesh)


# ==========================================================
# CAMERA SYSTEM
# ==========================================================

CAMERA_VIEWS = {
    "Isometric": ([0.8, -1.2, 0.6], [0, 0, 1]),
    "Front": ([0, -1, 0], [0, 0, 1]),
    "Side": ([1, 0, 0], [0, 0, 1]),
    "Top": ([0, 0, 1], [0, 1, 0]),
}


def render_with_camera(geometries, front, up, width=1200, height=900):

    if not isinstance(geometries, list):
        geometries = [geometries]

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=width, height=height)

    for g in geometries:
        vis.add_geometry(g)

    opt = vis.get_render_option()
    opt.background_color = np.array([1, 1, 1])
    opt.light_on = True
    opt.mesh_show_back_face = True

    ctr = vis.get_view_control()
    bbox = geometries[0].get_axis_aligned_bounding_box()
    center = bbox.get_center()

    ctr.set_lookat(center)
    ctr.set_front(front)
    ctr.set_up(up)
    ctr.set_zoom(0.7)

    vis.poll_events()
    vis.update_renderer()
    img = np.asarray(vis.capture_screen_float_buffer(False))
    vis.destroy_window()
    return img


# ==========================================================
# UNIFIED LEGEND BOX SYSTEM (SAFE + SELF-CONTAINED)
# ==========================================================

def draw_legend_box(ax, raw_text, fontsize=11, wrap_width=44):

    ax.axis("off")

    # ---- Layout controls ----
    left_pad = 0.08
    right_pad = 0.08
    top_pad = 0.08
    bottom_pad = 0.08

    text_left = left_pad + 0.03
    text_top = 1 - top_pad - 0.03

    # ---- Safe text wrapping ----
    wrapped_lines = []
    for paragraph in raw_text.split("\n"):
        if paragraph.strip() == "":
            wrapped_lines.append("")
        else:
            wrapped = textwrap.fill(paragraph, width=wrap_width)
            wrapped_lines.extend(wrapped.split("\n"))

    final_text = "\n".join(wrapped_lines)

    # ---- Dynamic height calculation ----
    line_height = 0.042
    text_height = line_height * len(wrapped_lines)

    box_height = text_height + 0.06
    max_height = 1 - top_pad - bottom_pad
    box_height = min(box_height, max_height)

    box_bottom = 1 - top_pad - box_height
    box_width = 1 - left_pad - right_pad

    rect = Rectangle(
        (left_pad, box_bottom),
        box_width,
        box_height,
        fill=False,
        linewidth=2,
        edgecolor="black",
        transform=ax.transAxes
    )
    ax.add_patch(rect)

    ax.text(
        text_left,
        text_top,
        final_text,
        va="top",
        fontsize=fontsize,
        linespacing=1.4,
        transform=ax.transAxes
    )


def save_with_legend(image, title, legend_text, output_path, cmap=None):

    fig = plt.figure(figsize=(18, 8))
    fig.patch.set_facecolor("white")

    ax_img = plt.axes([0.05, 0.1, 0.6, 0.8])
    ax_img.imshow(image, cmap=cmap)
    ax_img.set_title(title, fontsize=18, fontweight="bold")
    ax_img.axis("off")

    ax_leg = plt.axes([0.7, 0.1, 0.25, 0.8])
    draw_legend_box(ax_leg, legend_text, fontsize=11)

    plt.savefig(output_path, dpi=300)
    plt.close()


# ==========================================================
# CURVATURE
# ==========================================================

def compute_curvature_colors(mesh):

    mesh.compute_vertex_normals()
    normals = np.asarray(mesh.vertex_normals)

    mean_normal = normals.mean(axis=0)
    curvature = np.linalg.norm(normals - mean_normal, axis=1)

    min_val = float(curvature.min())
    max_val = float(curvature.max())
    denom = max(max_val - min_val, 1e-8)

    curvature = (curvature - min_val) / denom
    colors = plt.cm.inferno(curvature)[:, :3]

    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    return mesh


# ==========================================================
# NORMAL RGB
# ==========================================================

def compute_normal_colors(mesh):

    mesh.compute_vertex_normals()
    normals = np.asarray(mesh.vertex_normals)
    normals = (normals + 1.0) / 2.0

    mesh.vertex_colors = o3d.utility.Vector3dVector(normals)
    return mesh


# ==========================================================
# WIREFRAME
# ==========================================================

def compute_wireframe_overlay(mesh):

    mesh.paint_uniform_color([0.85, 0.85, 0.85])
    lines = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    lines.paint_uniform_color([0, 0, 0])

    return mesh, lines


# ==========================================================
# QUAD VIEW
# ==========================================================

def create_quad_view(mesh, output_path):

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor("white")

    view_items = list(CAMERA_VIEWS.items())

    for ax, (name, (front, up)) in zip(axes.flatten(), view_items):
        temp_mesh = clone_mesh(mesh)
        img = render_with_camera(temp_mesh, front, up)

        ax.imshow(img)
        ax.set_title(f"{name} View", fontsize=14, fontweight="bold")
        ax.axis("off")

    fig.add_artist(Line2D([0.5, 0.5], [0.05, 0.95],
                          transform=fig.transFigure,
                          linewidth=4, color="black"))

    fig.add_artist(Line2D([0.05, 0.95], [0.5, 0.5],
                          transform=fig.transFigure,
                          linewidth=4, color="black"))

    plt.savefig(output_path, dpi=300)
    plt.close()


# ==========================================================
# METRICS
# ==========================================================

def compute_architectural_metrics(mesh):

    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)

    bbox = mesh.get_axis_aligned_bounding_box()
    volume = np.prod(bbox.get_extent()) + 1e-8

    surface = min(len(verts) / volume / 10000, 1.0)

    mesh.compute_vertex_normals()
    normals = np.asarray(mesh.vertex_normals)
    geometric = 1.0 - min(np.var(normals) * 5, 1.0)

    cluster_result = mesh.cluster_connected_triangles()
    labels = np.asarray(cluster_result[0])

    if len(labels) > 0:
        largest = np.bincount(labels).max()
        continuity = largest / len(tris)
    else:
        continuity = 0.0

    tri_density = len(tris) / volume
    detail = min(tri_density / 20000, 1.0)

    topology = 1.0 - min(len(mesh.get_non_manifold_edges()) / 1000, 1.0)

    scores = np.clip(
        [surface, geometric, continuity, detail, topology],
        0, 1
    )

    return scores * 100


# ==========================================================
# SPIDER CHART
# ==========================================================

def create_spider_chart(scores, output_path):

    categories = [
        "Surface Completeness",
        "Geometric Consistency",
        "Structural Continuity",
        "Detail Resolution",
        "Topology Cleanliness",
    ]

    baseline = [55, 60, 50, 45, 65]

    N = len(categories)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    values = scores.tolist()
    values += values[:1]
    baseline += baseline[:1]

    fig = plt.figure(figsize=(16, 8))

    ax = plt.axes([0.05, 0.1, 0.55, 0.8], polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 100)

    ax.plot(angles, values, linewidth=3)
    ax.fill(angles, values, alpha=0.4)

    ax.plot(angles, baseline, linewidth=3)
    ax.fill(angles, baseline, alpha=0.2)

    ax.set_title("Architectural Quality Evaluation",
                 fontsize=18, fontweight="bold", pad=25)

    ax_leg = plt.axes([0.68, 0.1, 0.30, 0.8])

    legend_text = (
        "Surface Completeness: Coverage of mesh relative to "
        "volume. Higher means fewer missing regions.\n\n"
        "Geometric Consistency: Stability of surface normals. "
        "Higher means smoother, less noisy geometry.\n\n"
        "Structural Continuity: Connectivity of mesh components. "
        "Higher means minimal fragmentation.\n\n"
        "Detail Resolution: Triangle density per volume. "
        "Higher means finer geometric detail captured.\n\n"
        "Topology Cleanliness: Presence of invalid or "
        "non-manifold edges. Higher means structurally valid "
        "mesh topology.\n\n"
        "Interpretation: Higher overall scores reflect a more "
        "complete, stable, detailed, and structurally sound "
        "reconstruction."
    )

    draw_legend_box(ax_leg, legend_text, fontsize=10)

    plt.savefig(output_path, dpi=300)
    plt.close()


# ==========================================================
# MAIN
# ==========================================================

def run(run_root: Path, project_root: Path, force: bool, logger):

    paths = ProjectPaths(project_root)
    load_config(run_root)

    mesh_path = paths.mesh / "mesh_cleaned.ply"
    vis_dir = paths.visualization
    vis_dir.mkdir(parents=True, exist_ok=True)

    if not mesh_path.exists():
        logger.info("[vis] No mesh found")
        return

    logger.info("[vis] Loading mesh")
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    mesh.compute_vertex_normals()

    logger.info("[vis] Creating quad composite")
    create_quad_view(mesh, vis_dir / "multi_view_composite.png")

    logger.info("[vis] Creating curvature heatmap")
    curv_mesh = compute_curvature_colors(clone_mesh(mesh))
    img = render_with_camera(curv_mesh, *CAMERA_VIEWS["Isometric"])

    save_with_legend(
        img,
        "Curvature Heatmap",
        "Curvature via normal deviation (normalized 0–1).\n\n"
        "Bright → high surface variation\n"
        "Dark → planar/smooth regions",
        vis_dir / "curvature_heatmap.png"
    )

    logger.info("[vis] Creating normal visualization")
    norm_mesh = compute_normal_colors(clone_mesh(mesh))
    img = render_with_camera(norm_mesh, *CAMERA_VIEWS["Isometric"])

    save_with_legend(
        img,
        "Normal RGB Visualization",
        "Normals encoded as RGB.\n"
        "R = X | G = Y | B = Z\n\n"
        "Abrupt shifts → orientation instability\n"
        "Smooth gradients → stable geometry",
        vis_dir / "normal_visualization.png"
    )

    logger.info("[vis] Creating wireframe overlay")
    shaded, lines = compute_wireframe_overlay(clone_mesh(mesh))
    img = render_with_camera([shaded, lines], *CAMERA_VIEWS["Isometric"])

    save_with_legend(
        img,
        "Wireframe Overlay",
        "Black lines = triangle edges.\n\n"
        "Dense → high detail\n"
        "Sparse → low resolution\n"
        "Irregular → topology artifacts",
        vis_dir / "wireframe_overlay.png"
    )

    logger.info("[vis] Creating spider chart")
    scores = compute_architectural_metrics(mesh)
    create_spider_chart(scores, vis_dir / "architectural_spider_chart.png")

    logger.info("[vis] Visualization suite complete")