#!/usr/bin/env python3
"""
image_analyzer.py

MARK-2 Dataset Intelligence Stage (Input-Aware)
----------------------------------------------
Consumes images produced by input.py and emits
config-ready dataset intelligence.

INPUT  : project_root/images_processed/
OUTPUT : run_root/evaluation/dataset_intelligence.json
"""

from pathlib import Path
import json
import cv2
import numpy as np
from collections import Counter

from utils.paths import ProjectPaths

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

# ----------------------------
# Low-level metrics
# ----------------------------
def compute_blur(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def compute_luminance(gray):
    return float(gray.mean()), float(gray.std())

def compute_clipping(gray, low=5, high=250):
    total = gray.size
    return float(np.sum(gray < low) / total), float(np.sum(gray > high) / total)

def compute_orb_features(gray, max_features=4000):
    orb = cv2.ORB_create(nfeatures=max_features)
    return len(orb.detect(gray, None))

def perceptual_hash(gray, size=8):
    resized = cv2.resize(gray, (size * 4, size * 4))
    dct = cv2.dct(np.float32(resized))
    block = dct[:size, :size]
    med = np.median(block)
    return (block > med).flatten()

def hamming(a, b):
    return int(np.sum(a != b))

# ----------------------------
# Dataset helpers
# ----------------------------
def dataset_scale(n):
    if n < 20:
        return "invalid"
    if n < 50:
        return "tiny"
    if n < 300:
        return "small"
    if n < 2000:
        return "medium"
    if n < 10000:
        return "large"
    return "massive"

def suggest_downsample(max_dim, target=1600):
    return float(min(1.0, target / max_dim))

# ----------------------------
# Pipeline stage
# ----------------------------
def run(run_root: Path, project_root: Path, force: bool, logger):
    run_root = run_root.resolve()
    project_root = project_root.resolve()

    project_paths = ProjectPaths(project_root)
    img_dir = project_paths.images_processed

    eval_dir = run_root / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    out_json = eval_dir / "dataset_intelligence.json"
    if out_json.exists() and not force:
        logger.info("[analyzer] dataset_intelligence.json exists — skipping")
        return

    if not img_dir.exists():
        raise RuntimeError("images_processed directory missing — input stage failed")

    images = [p for p in sorted(img_dir.iterdir()) if p.suffix.lower() in SUPPORTED_IMAGE_EXTS]
    if not images:
        raise RuntimeError("No images found in images_processed/")

    logger.info(f"[analyzer] Processing {len(images)} images from {img_dir}")

    # ----------------------------
    # Accumulators
    # ----------------------------
    widths, heights = [], []
    blur_vals = []
    lum_means, lum_stds = [], []
    low_clip, high_clip = [], []
    feat_counts, feat_density = [], []
    aspect_ratios = []
    orientations = Counter()
    hashes = []

    sample_step = max(1, len(images) // 300)

    # ----------------------------
    # Per-image analysis
    # ----------------------------
    for idx, img_path in enumerate(images, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning(f"[analyzer] Failed to read {img_path.name}")
            continue

        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        widths.append(w)
        heights.append(h)
        aspect_ratios.append(w / h)
        orientations["landscape" if w >= h else "portrait"] += 1

        blur_vals.append(compute_blur(gray))
        lm, ls = compute_luminance(gray)
        lum_means.append(lm)
        lum_stds.append(ls)

        lo, hi = compute_clipping(gray)
        low_clip.append(lo)
        high_clip.append(hi)

        feats = compute_orb_features(gray)
        feat_counts.append(feats)
        feat_density.append(feats / ((w * h) / 1e6))

        if idx % sample_step == 0:
            hashes.append(perceptual_hash(gray))

        if idx % 100 == 0:
            logger.info(f"[analyzer] Processed {idx}/{len(images)} images")

    # ----------------------------
    # Overlap analysis
    # ----------------------------
    similarities = []
    orphan_count = 0
    for i, h1 in enumerate(hashes):
        dists = [hamming(h1, h2) for j, h2 in enumerate(hashes) if i != j]
        if not dists:
            continue
        nearest = min(dists)
        similarities.append(nearest)
        if nearest > 20:
            orphan_count += 1

    # ----------------------------
    # Aggregate intelligence
    # ----------------------------
    max_dim = max(max(widths), max(heights))

    intelligence = {
        "image_count": len(images),
        "dataset_scale": dataset_scale(len(images)),
        "resolution": {
            "min": [int(min(widths)), int(min(heights))],
            "max": [int(max(widths)), int(max(heights))],
            "mean": [int(np.mean(widths)), int(np.mean(heights))],
        },
        "quality": {
            "blur": {
                "mean": float(np.mean(blur_vals)) if blur_vals else 0.0,
                "low_ratio": float(np.mean(np.array(blur_vals) < 100)) if blur_vals else 0.0,
            },
            "luminance": {
                "mean": float(np.mean(lum_means)) if lum_means else 0.0,
                "std": float(np.mean(lum_stds)) if lum_stds else 0.0,
                "low_clip_ratio": float(np.mean(low_clip)) if low_clip else 0.0,
                "high_clip_ratio": float(np.mean(high_clip)) if high_clip else 0.0,
            },
        },
        "features": {
            "mean_count": int(np.mean(feat_counts)) if feat_counts else 0,
            "mean_per_megapixel": float(np.mean(feat_density)) if feat_density else 0.0,
            "low_density_ratio": float(np.mean(np.array(feat_density) < 300)) if feat_density else 0.0,
        },
        "overlap": {
            "mean_hamming_distance": float(np.mean(similarities)) if similarities else None,
            "orphan_ratio": orphan_count / max(1, len(similarities)) if similarities else 0.0,
        },
        "viewpoint": {
            "aspect_ratio_std": float(np.std(aspect_ratios)) if aspect_ratios else 0.0,
        },
        "orientation": dict(orientations),
        "downsample_factor": suggest_downsample(max_dim),
        "flags": [],
        "recommendations": [],
    }

    # ----------------------------
    # Flags
    # ----------------------------
    if intelligence["image_count"] < 20:
        intelligence["flags"].append("INSUFFICIENT_IMAGES")
    if intelligence["quality"]["blur"]["low_ratio"] > 0.4:
        intelligence["flags"].append("HIGH_BLUR_RATIO")
    if intelligence["features"]["low_density_ratio"] > 0.5:
        intelligence["flags"].append("LOW_FEATURE_DENSITY")
    if intelligence["overlap"]["orphan_ratio"] > 0.3:
        intelligence["flags"].append("POOR_OVERLAP")
    if intelligence["dataset_scale"] in {"large", "massive"}:
        intelligence["recommendations"].append("Enable batch mapping")

    # ----------------------------
    # Write output
    # ----------------------------
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(intelligence, f, indent=2)

    logger.info(f"[analyzer] Dataset intelligence written: {out_json}")
    logger.info(f"[analyzer] Flags: {intelligence['flags']}")
    logger.info("[analyzer] MARK-2 Dataset Intelligence completed")
