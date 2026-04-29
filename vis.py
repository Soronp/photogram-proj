#!/usr/bin/env python3
"""
MARK-2 Visualization (v27 - Clean Rewrite)
------------------------------------------
✔ Upright pentagon radar preserved
✔ Legend safely pushed below (no overlap guaranteed)
✔ Cleaner modular structure
✔ Stable color + normalization system
✔ Paper-ready outputs
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

DPI = 300


# =====================================================
# 🎨 COLOR SYSTEM
# =====================================================
COLOR_MAP = {
    "colmap": "#1f77b4",
    "openmvs": "#2ca02c",
    "openmvg": "#ff7f0e",
    "nerf": "#d62728",
    "hybrid": "#9467bd",
    "default": "#4c4c4c"
}


def get_color(name: str) -> str:
    name = name.lower()
    for k, v in COLOR_MAP.items():
        if k in name:
            return v
    return COLOR_MAP["default"]


def build_legend(models):
    return [
        Line2D(
            [0], [0],
            marker='o',
            color='w',
            label=m,
            markerfacecolor=get_color(m),
            markersize=8
        )
        for m in models
    ]


# =====================================================
# IO
# =====================================================
def load_json(path: str):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    return json.loads(path.read_text(encoding="utf-8"))


def get_models(data):
    return data["per_model_metrics"]


# =====================================================
# FEATURE SPACE
# =====================================================
def build_axes(m):
    return {
        "Accuracy": 1.0 / (1.0 + m.get("accuracy_mean", 1.0)),
        "Completeness": 1.0 / (1.0 + m.get("completeness_mean", 1.0)),
        "Surface": 1.0 / (1.0 + m.get("chamfer_distance", 1.0)),
        "ICP": m.get("icp_fitness", 0.0),
        "Coverage": m.get("coverage_ratio", 0.0),
    }


def normalize(data):
    keys = list(next(iter(data.values())).keys())
    out = {m: {} for m in data}

    for k in keys:
        vals = np.array([data[m][k] for m in data], dtype=float)

        vmin, vmax = vals.min(), vals.max()

        if abs(vmax - vmin) < 1e-8:
            scaled = np.full_like(vals, 0.5)
        else:
            scaled = (vals - vmin) / (vmax - vmin)

        for i, m in enumerate(data):
            out[m][k] = float(scaled[i])

    return out


def sort_models(models):
    return sorted(
        models.keys(),
        key=lambda m: models[m].get("fscore", 0.0),
        reverse=True
    )


# =====================================================
# RADAR GEOMETRY
# =====================================================
def radar_xy(values):
    n = len(values)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + np.pi / 2

    x = values * np.cos(angles)
    y = values * np.sin(angles)

    return np.append(x, x[0]), np.append(y, y[0])


def draw_grid(ax, n):
    base = np.linspace(0, 2 * np.pi, n, endpoint=False) + np.pi / 2

    for r in [0.2, 0.4, 0.6, 0.8]:
        xs = r * np.cos(base)
        ys = r * np.sin(base)
        ax.plot(np.append(xs, xs[0]), np.append(ys, ys[0]),
                color="#777", alpha=0.35, linewidth=0.9)

    xs = np.cos(base)
    ys = np.sin(base)
    ax.plot(np.append(xs, xs[0]), np.append(ys, ys[0]),
            color="#222", linewidth=2.4)

    for a in base:
        ax.plot([0, np.cos(a)], [0, np.sin(a)],
                color="#888", alpha=0.25, linewidth=0.8)


# =====================================================
# 🕸️ RADAR PLOT (ONLY LEGEND POSITION ADJUSTED)
# =====================================================
def plot_radar(per_model, out_path):
    models = sort_models(per_model)

    semantic = {m: build_axes(per_model[m]) for m in models}
    norm = normalize(semantic)

    labels = list(next(iter(norm.values())).keys())
    angles = len(labels)

    fig, ax = plt.subplots(figsize=(9, 9), facecolor="#f5f1ea")
    ax.set_facecolor("#f5f1ea")
    ax.set_aspect("equal")

    draw_grid(ax, angles)

    base_angles = np.linspace(0, 2 * np.pi, angles, endpoint=False) + np.pi / 2

    for m in models:
        vals = np.array(list(norm[m].values()))
        x, y = radar_xy(vals)

        c = get_color(m)
        ax.plot(x, y, linewidth=2, color=c)
        ax.fill(x, y, alpha=0.12, color=c)

    for i, lab in enumerate(labels):
        ax.text(
            1.15 * np.cos(base_angles[i]),
            1.15 * np.sin(base_angles[i]),
            lab,
            ha="center",
            va="center",
            fontsize=11
        )

    for r in [0.2, 0.4, 0.6, 0.8, 1.0]:
        ax.text(0, r, f"{r:.1f}", fontsize=9, ha="center", color="#444")

    ax.axis("off")

    plt.title(
        "Reconstruction Quality (MARK-2 v27)",
        fontsize=14,
        fontweight="bold",
        pad=20
    )

    # 🔥 ONLY CHANGE: moved legend further down
    fig.legend(
        handles=build_legend(models),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),   # was -0.02 → now lower
        ncol=3,
        frameon=False
    )

    plt.subplots_adjust(bottom=0.12)  # slightly more breathing room

    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close()


# =====================================================
# BAR PLOT (UNCHANGED)
# =====================================================
def plot_fscore(per_model, out_path):
    models = sort_models(per_model)

    scores = [per_model[m].get("fscore", 0) for m in models]
    colors = [get_color(m) for m in models]

    fig, ax = plt.subplots(figsize=(8, 4), facecolor="#f5f1ea")

    ax.bar(models, scores, color=colors, edgecolor="#333", linewidth=0.5)
    ax.set_title("F-score Ranking")
    ax.set_ylabel("Score")
    ax.grid(axis="y", alpha=0.2)

    plt.xticks(rotation=15)
    plt.tight_layout()

    plt.savefig(out_path, dpi=DPI)
    plt.close()


# =====================================================
# ICP PLOT (UNCHANGED)
# =====================================================
def plot_icp(per_model, out_path):
    fig, ax = plt.subplots(figsize=(6, 5), facecolor="#f5f1ea")

    for m, v in per_model.items():
        ax.scatter(
            v.get("icp_rmse", 0),
            v.get("icp_fitness", 0),
            s=120,
            color=get_color(m),
            edgecolor="#333"
        )

    ax.set_title("ICP Alignment Quality")
    ax.set_xlabel("ICP RMSE")
    ax.set_ylabel("ICP Fitness")
    ax.grid(alpha=0.25)

    fig.legend(
        handles=build_legend(per_model.keys()),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),  # moved slightly lower
        ncol=3,
        frameon=False
    )

    plt.subplots_adjust(bottom=0.12)
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close()


# =====================================================
# TRADEOFF PLOT (UNCHANGED)
# =====================================================
def plot_tradeoff(per_model, out_path):
    fig, ax = plt.subplots(figsize=(6, 5), facecolor="#f5f1ea")

    for m, v in per_model.items():
        ax.scatter(
            v.get("accuracy_mean", 0),
            v.get("coverage_ratio", 0),
            s=120,
            color=get_color(m),
            edgecolor="#333"
        )

    ax.set_title("Accuracy–Coverage Tradeoff")
    ax.set_xlabel("Accuracy")
    ax.set_ylabel("Coverage")
    ax.grid(alpha=0.25)

    fig.legend(
        handles=build_legend(per_model.keys()),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),  # moved slightly lower
        ncol=3,
        frameon=False
    )

    plt.subplots_adjust(bottom=0.12)
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close()


# =====================================================
# PIPELINE
# =====================================================
def run(path):
    data = load_json(path)
    per_model = get_models(data)

    out_dir = Path(path).parent / "viz_v27"
    out_dir.mkdir(exist_ok=True)

    print(f"\n[INFO] Models: {len(per_model)}")
    print(f"[INFO] Output: {out_dir}")

    plot_radar(per_model, out_dir / "radar.png")
    plot_fscore(per_model, out_dir / "fscore.png")
    plot_icp(per_model, out_dir / "icp.png")
    plot_tradeoff(per_model, out_dir / "tradeoff.png")

    print("[INFO] Done.")


# =====================================================
# ENTRY
# =====================================================
if __name__ == "__main__":
    print("\n=== MARK-2 Visualization v27 (Adjusted Legend) ===")
    p = input("Enter results.json path: ").strip()

    if not Path(p).exists():
        raise FileNotFoundError(p)

    run(p)