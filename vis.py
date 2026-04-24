#!/usr/bin/env python3
"""
MARK-2 Visualization (Robust V3 - Interpretable)
------------------------------------------------
Enhancements:
✔ Larger spider chart (readable for many models)
✔ Explicit legends placed outside (no overlap)
✔ Auto scaling (linear/log)
✔ Clear semantic labeling of metrics
✔ Color meaning explained in titles
✔ Consistent evaluator schema
"""

import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False


# =====================================================
# LOAD
# =====================================================
def load_results(path):
    with open(path, "r") as f:
        return json.load(f)


# =====================================================
# AUTO SCALE
# =====================================================
def auto_scale(ax, values, ylabel):
    if not values:
        return "linear"

    v = np.array(values, dtype=float)
    v = v[np.isfinite(v)]

    if len(v) == 0:
        return "linear"

    ratio = (np.max(v) + 1e-12) / (np.min(v) + 1e-12)

    if ratio > 1e3:
        ax.set_yscale("log")
        ax.set_ylabel(ylabel + " (log scale)")
        return "log"
    else:
        ax.set_ylabel(ylabel + " (linear scale)")
        return "linear"


# =====================================================
# NORMALIZE
# =====================================================
def normalize(d, invert=False):
    if not d:
        return {}

    vals = np.array(list(d.values()), dtype=float)
    mn, mx = np.min(vals), np.max(vals)

    if abs(mx - mn) < 1e-8:
        return {k: 0.5 for k in d}

    out = {}
    for k, v in d.items():
        x = (v - mn) / (mx - mn)
        out[k] = 1 - x if invert else x
    return out


# =====================================================
# BUILD MATRIX
# =====================================================
def build_matrix(pairwise, key):
    models = sorted({m for k in pairwise for m in k.split("__vs__")})
    idx = {m: i for i, m in enumerate(models)}
    mat = np.zeros((len(models), len(models)))

    for k, v in pairwise.items():
        if key not in v:
            continue
        a, b = k.split("__vs__")
        i, j = idx[a], idx[b]
        mat[i, j] = mat[j, i] = float(v[key])

    return models, mat


# =====================================================
# HEATMAP
# =====================================================
def plot_heatmap(models, matrix, title, out):
    if len(models) == 0:
        return

    plt.figure(figsize=(12, 10))

    if SEABORN_AVAILABLE:
        sns.heatmap(matrix,
                    xticklabels=models,
                    yticklabels=models,
                    cmap="viridis",
                    square=True,
                    cbar_kws={"label": "Distance (lower = better)"})
    else:
        im = plt.imshow(matrix)
        plt.colorbar(im, label="Distance (lower = better)")
        plt.xticks(range(len(models)), models, rotation=90)
        plt.yticks(range(len(models)), models)

    plt.title(title + "\nColor Meaning: Dark = Similar (Good), Bright = Dissimilar (Bad)")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


# =====================================================
# BAR
# =====================================================
def plot_bar(data, ylabel, title, out, better="lower"):
    if not data:
        return

    names = list(data.keys())
    values = list(data.values())

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(names, values)
    plt.xticks(rotation=90)

    scale_type = auto_scale(ax, values, ylabel)

    meaning = "Lower = Better" if better == "lower" else "Higher = Better"
    ax.set_title("Multi-Metric Comparison", pad=20)

    # Legend block (outside)
    ax.legend([f"Metric scale: {scale_type}"], loc="center left", bbox_to_anchor=(1, 0.5))

    plt.tight_layout(rect=[0, 0, 0.8, 1])
    plt.savefig(out)
    plt.close()


# =====================================================
# SPIDER (LARGER + CLEAR)
# =====================================================
def plot_spider(metrics, out):
    if not metrics:
        return

    labels = list(next(iter(metrics.values())).keys())
    n = len(labels)

    angles = np.linspace(0, 2*np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(12, 12))  # LARGER
    ax = plt.subplot(111, polar=True)

    for name, vals in metrics.items():
        v = list(vals.values())
        v += v[:1]
        ax.plot(angles, v, label=name)
        ax.fill(angles, v, alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)

    ax.set_title(
        "Multi-Metric Comparison\n"
        "Interpretation:\n"
        "- All values normalized to [0,1]\n"
        "- Higher = Better (after inversion where needed)\n"
        "- Larger area = better overall model",
        pad=30
    )

    ax.legend(loc="center left", bbox_to_anchor=(1.2, 0.5))

    plt.tight_layout(rect=[0, 0, 0.75, 1])
    plt.savefig(out)
    plt.close()


# =====================================================
# MAIN
# =====================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    data = load_results(args.input)

    pairwise = data.get("pairwise", {})
    per_model = data.get("per_model", {})

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Heatmaps
    models, chamfer = build_matrix(pairwise, "chamfer_mean")
    _, icp = build_matrix(pairwise, "icp_rmse")

    plot_heatmap(models, chamfer, "Chamfer Distance Heatmap", out / "chamfer.png")
    plot_heatmap(models, icp, "ICP RMSE Heatmap", out / "icp.png")

    # Bars
    dispersion = {m: v.get("projection_dispersion", 0) for m, v in per_model.items()}
    points = {m: v.get("num_points", 0) for m, v in per_model.items()}

    plot_bar(dispersion, "Projection Dispersion", "Projection Stability", out / "dispersion.png", better="lower")
    plot_bar(points, "Point Count", "Reconstruction Density", out / "points.png", better="higher")

    # Spider
    chamfer_avg = {m: float(np.mean(chamfer[i])) for i, m in enumerate(models)} if len(models) else {}
    icp_avg = {m: float(np.mean(icp[i])) for i, m in enumerate(models)} if len(models) else {}

    c_n = normalize(chamfer_avg, invert=True)
    i_n = normalize(icp_avg, invert=True)
    d_n = normalize(dispersion, invert=True)

    common = set(c_n) & set(i_n) & set(d_n)

    spider = {
        m: {
            "Chamfer (shape similarity)": c_n[m],
            "ICP (alignment quality)": i_n[m],
            "Dispersion (projection stability)": d_n[m]
        }
        for m in common
    }

    plot_spider(spider, out / "spider.png")

    print(f"[INFO] Saved visualizations → {out}")


if __name__ == "__main__":
    main()
