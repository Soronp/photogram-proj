#!/usr/bin/env python3
"""
image_analyzer.py

MARK-2 Dataset Analyzer Stage
----------------------------
Analyzes raw images and produces diagnostics.
"""

from pathlib import Path
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt

from utils.paths import ProjectPaths

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


# --------------------------------------------------
# METRICS
# --------------------------------------------------

def compute_blur(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_features(image, max_features=5000):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=max_features)
    return int(len(orb.detect(gray, None)))


def suggest_downsample(widths, heights, target_max=1600):
    return float(min(1.0, target_max / max(max(widths), max(heights))))


# --------------------------------------------------
# PIPELINE STAGE
# --------------------------------------------------

def run(project_root: Path, force: bool, logger):
    paths = ProjectPaths(project_root)
    paths.ensure_all()

    raw_dir = paths.raw
    eval_dir = paths.evaluation
    eval_dir.mkdir(parents=True, exist_ok=True)

    logger.info("[analyzer] Stage started")

    images = [
        p for p in sorted(raw_dir.rglob("*"))
        if p.suffix.lower() in SUPPORTED_IMAGE_EXTS
    ]

    if not images:
        logger.warning("[analyzer] No images found in raw/")
        return

    widths, heights = [], []
    blur_scores, feature_counts, fpm = [], [], []

    diagnostics = {
        "image_count": len(images),
        "resolutions": [],
        "blur_scores": [],
        "feature_counts": [],
        "features_per_megapixel": [],
        "recommendations": [],
    }

    for path in images:
        img = cv2.imread(str(path))
        if img is None:
            continue

        h, w = img.shape[:2]
        widths.append(w)
        heights.append(h)

        blur = compute_blur(img)
        feats = compute_features(img)
        per_mp = feats / ((w * h) / 1e6)

        blur_scores.append(blur)
        feature_counts.append(feats)
        fpm.append(per_mp)

        diagnostics["resolutions"].append({"width": w, "height": h})
        diagnostics["blur_scores"].append(blur)
        diagnostics["feature_counts"].append(feats)
        diagnostics["features_per_megapixel"].append(per_mp)

    diagnostics.update({
        "avg_blur": float(np.mean(blur_scores)),
        "avg_features": int(np.mean(feature_counts)),
        "avg_features_per_megapixel": float(np.mean(fpm)),
        "downsample_factor": suggest_downsample(widths, heights),
    })

    if diagnostics["avg_blur"] < 100:
        diagnostics["recommendations"].append("Low average blur detected")

    if diagnostics["avg_features"] < 1000:
        diagnostics["recommendations"].append("Low feature density detected")

    out_json = eval_dir / "dataset_diagnostics.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)

    logger.info(f"[analyzer] Diagnostics written: {out_json}")

    # --------------------------------------------------
    # Plots (best-effort)
    # --------------------------------------------------

    try:
        plt.hist(feature_counts, bins=20)
        plt.title("Feature Count Distribution")
        plt.tight_layout()
        plt.savefig(eval_dir / "feature_distribution.png")
        plt.close()
    except Exception as e:
        logger.warning(f"[analyzer] Plot generation failed: {e}")

    logger.info("[analyzer] Stage completed")
