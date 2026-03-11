#!/usr/bin/env python3
"""
MARK-2 Scientific Visualization Stage (Thesis Edition)
------------------------------------------------------

Clear, non-overlapping, publication-grade figures for
Architectural Value Extraction Framework.

Each figure includes:
    • Architectural meaning in subtitle
    • Improved spacing (no collisions)
    • Reduced annotation clutter
    • Vector export (.pdf) + high DPI PNG
    • Auto-generated captions.txt
"""

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================================
# GLOBAL STYLE
# ==========================================================

plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.family": "serif",
    "axes.titlesize": 13,
    "axes.labelsize": 12
})

sns.set_style("whitegrid")


# ==========================================================
# UTIL: SAVE FIGURE (PNG + PDF)
# ==========================================================

def save_figure(fig, out_path: Path):
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


# ==========================================================
# 1️⃣ ARCHITECTURAL METRICS BAR
# ==========================================================

def save_metrics_bar(metrics, out_path):
    keys = [
        "geometric_fidelity",
        "structural_regularity",
        "surface_integrity",
        "detail_richness",
        "spatial_coherence"
    ]

    labels = [
        "Geometric Fidelity",
        "Structural Regularity",
        "Surface Integrity",
        "Detail Richness",
        "Spatial Coherence"
    ]

    values = [metrics.get(k, 0.0) for k in keys]
    avi = metrics.get("architectural_value_index", 0.0)

    fig, ax = plt.subplots(figsize=(10,5), constrained_layout=True)
    bars = ax.bar(labels, values)

    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Normalized Score")
    ax.set_title(
        f"Architectural Value Profile\n"
        f"Composite Index (AVI) = {avi:.3f}",
        pad=12
    )

    ax.tick_params(axis='x', rotation=20)

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2,
            height + 0.03,
            f"{height:.2f}",
            ha='center',
            fontsize=10
        )

    save_figure(fig, out_path)


# ==========================================================
# 2️⃣ STRUCTURAL REGULARITY
# ==========================================================

def save_structural_regularity(data, out_path):
    arr = np.array(data)
    if len(arr) == 0:
        return

    fig, ax = plt.subplots(figsize=(7,5), constrained_layout=True)
    sns.boxplot(y=arr, ax=ax)

    mean_val = np.mean(arr)
    ax.set_title(
        "Structural Regularity — Wall Planarity Consistency\n"
        f"Mean Variance = {mean_val:.5e}  (Lower = Straighter Walls)",
        pad=10
    )
    ax.set_ylabel("Plane Inlier Variance")

    save_figure(fig, out_path)


# ==========================================================
# 3️⃣ DETAIL RICHNESS
# ==========================================================

def save_detail_richness(data, out_path):
    arr = np.array(data)
    if len(arr) == 0:
        return

    if len(arr) > 80_000:
        idx = np.random.choice(len(arr), 80_000, replace=False)
        arr = arr[idx]

    fig, ax = plt.subplots(figsize=(7,5), constrained_layout=True)
    sns.violinplot(y=arr, inner="quartile", ax=ax)

    ax.set_title(
        "Detail Richness — Surface Ornamentation Intensity\n"
        "(Higher Curvature = More Architectural Detail)",
        pad=10
    )
    ax.set_ylabel("Mean Curvature Magnitude")

    save_figure(fig, out_path)


# ==========================================================
# 4️⃣ SPATIAL COHERENCE (FIXED CLEAN VERSION)
# ==========================================================

def save_spatial_coherence(data, out_path):
    arr = np.array(data)
    if len(arr) == 0:
        return

    mean_val = np.mean(arr)
    std_val = np.std(arr)
    cv = std_val / (mean_val + 1e-12)

    fig, ax = plt.subplots(figsize=(8,5), constrained_layout=True)
    sns.boxplot(y=arr, ax=ax)

    # Expand limits to prevent text overlap
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 1.1)

    ax.set_title(
        "Spatial Coherence — Reconstruction Density Uniformity\n"
        f"Coefficient of Variation (CV) = {cv:.4f}  (Lower = More Uniform)",
        pad=12
    )
    ax.set_ylabel("Normalized Local Vertex Density")

    save_figure(fig, out_path)


# ==========================================================
# 5️⃣ SURFACE INTEGRITY
# ==========================================================

def save_surface_integrity(topo, out_path):
    keys = ["non_manifold_edges", "boundary_edges", "components"]
    labels = ["Non-Manifold Edges", "Boundary Edges", "Connected Components"]
    values = [topo.get(k, 0) for k in keys]

    fig, ax = plt.subplots(figsize=(9,4.5), constrained_layout=True)
    bars = ax.bar(labels, values)

    ax.set_ylabel("Count")
    ax.set_title(
        "Surface Integrity — Topological Defects\n"
        "(Mesh Completeness and Connectivity)",
        pad=10
    )

    ax.tick_params(axis='x', rotation=15)

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2,
            height + max(values)*0.02 if max(values) > 0 else 0.1,
            f"{int(height)}",
            ha='center',
            fontsize=10
        )

    save_figure(fig, out_path)


# ==========================================================
# 6️⃣ CURVATURE–DENSITY HEXBIN
# ==========================================================

def save_hexbin(curvature, density, out_path):
    if not curvature or not density:
        return

    N = min(len(curvature), len(density), 80_000)
    idx = np.random.choice(len(curvature), N, replace=False)

    curv = np.array(curvature)[idx]
    dens = np.array(density)[idx]

    fig, ax = plt.subplots(figsize=(8,6), constrained_layout=True)

    hb = ax.hexbin(curv, dens, gridsize=40, bins='log')
    fig.colorbar(hb, ax=ax, label="Log Density")

    ax.set_xlabel("Curvature")
    ax.set_ylabel("Local Density")

    ax.set_title(
        "Curvature–Density Interaction\n"
        "(Relationship Between Ornamentation and Reconstruction Coverage)",
        pad=12
    )

    save_figure(fig, out_path)


# ==========================================================
# CAPTION GENERATOR
# ==========================================================

def generate_captions(vis_dir):
    captions = """
Figure 1: Architectural Value Profile.
Normalized scores for five architectural dimensions and composite index (AVI).

Figure 2: Structural Regularity.
Distribution of RANSAC plane inlier variance measuring wall and surface planarity.

Figure 3: Detail Richness.
Curvature magnitude distribution indicating architectural ornamentation intensity.

Figure 4: Spatial Coherence.
Local density uniformity; lower CV indicates more evenly reconstructed geometry.

Figure 5: Surface Integrity.
Counts of non-manifold edges, boundary edges, and disconnected components.

Figure 6: Curvature–Density Interaction.
Density map showing relationship between geometric detail and reconstruction coverage.
"""
    with open(vis_dir / "captions.txt", "w") as f:
        f.write(captions.strip())


# ==========================================================
# MAIN ENTRY
# ==========================================================

def run(run_root: Path, project_root: Path, input_path: Path, force: bool, logger):

    from utils.paths import ProjectPaths
    paths = ProjectPaths(project_root)
    paths.ensure_all()

    eval_path = paths.evaluation / "architectural_metrics.json"
    vis_dir = paths.visualization
    vis_dir.mkdir(parents=True, exist_ok=True)

    if not eval_path.exists():
        logger.error(f"[vis] Evaluation metrics not found at {eval_path}")
        return

    with open(eval_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    metrics = data.get("architectural_metrics", {})
    diagnostics = data.get("diagnostics", {})

    save_metrics_bar(metrics, vis_dir / "architectural_metrics_bar.png")
    save_structural_regularity(diagnostics.get("plane_variances", []),
                               vis_dir / "structural_regularity_box.png")
    save_detail_richness(diagnostics.get("curvature", []),
                         vis_dir / "detail_richness_violin.png")
    save_spatial_coherence(diagnostics.get("density", []),
                           vis_dir / "spatial_coherence_box.png")
    save_surface_integrity(diagnostics.get("topology", {}),
                           vis_dir / "surface_integrity_bar.png")
    save_hexbin(diagnostics.get("curvature", []),
                diagnostics.get("density", []),
                vis_dir / "curvature_vs_density_hexbin.png")

    generate_captions(vis_dir)

    logger.info(f"[vis] MARK-2 Visualization Complete → {vis_dir}")