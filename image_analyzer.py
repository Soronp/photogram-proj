#!/usr/bin/env python3
"""
image_analyzer.py

MARK-2 Dataset Analyzer + Preprocessing Controller
--------------------------------------------------
- Analyzes images in raw/ (output of input.py)
- Provides dataset diagnostics and recommendations
- Produces filtering and downsampling instructions
- Generates JSON, plots, and logs

Stage order:
    input -> image_analyzer -> image_filter -> pre_processing
"""

from pathlib import Path
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.logger import get_logger
from utils.paths import ProjectPaths

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

# --------------------------------------------------
# Image metrics
# --------------------------------------------------

def compute_blur_score(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def compute_feature_density(image: np.ndarray, max_features: int = 5000) -> int:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=max_features)
    kp = orb.detect(gray, None)
    return int(len(kp))

def suggest_downsample(widths, heights, target_max: int = 1600) -> float:
    max_w = max(widths)
    max_h = max(heights)
    return float(min(1.0, target_max / max(max_w, max_h)))

# --------------------------------------------------
# Core analyzer
# --------------------------------------------------

def run(project_root: Path, force: bool = False, detailed_json: bool = True):
    paths = ProjectPaths(project_root)
    paths.ensure_all()

    logger = get_logger("image_analyzer", project_root)
    logger.info("Starting dataset analysis (raw images)")

    raw_dir = paths.raw
    images = [
        p for p in sorted(raw_dir.rglob("*"))
        if p.suffix.lower() in SUPPORTED_IMAGE_EXTS
    ]

    if not images:
        logger.warning("No images found in raw/ for analysis")
        return None

    diagnostics = {
        "image_count": int(len(images)),
        "resolutions": [],
        "aspect_ratios": [],
        "blur_scores": [],
        "feature_counts": [],
        "features_per_megapixel": [],
        "filter_flags": [],
        "preprocessing": {},
        "recommendations": [],
        "detailed_per_image": [] if detailed_json else None,
    }

    widths, heights = [], []
    blur_scores, feature_counts, features_per_mp = [], [], []

    for img_path in tqdm(images, desc="Analyzing raw images"):
        img = cv2.imread(str(img_path))

        if img is None:
            logger.warning(f"Unreadable image: {img_path.name}")
            diagnostics["filter_flags"].append(True)
            if detailed_json:
                diagnostics["detailed_per_image"].append({
                    "file": img_path.name,
                    "error": "Unreadable"
                })
            continue

        h, w = img.shape[:2]
        widths.append(int(w))
        heights.append(int(h))

        blur = compute_blur_score(img)
        feat_count = compute_feature_density(img)
        feat_per_mp = feat_count / ((w * h) / 1e6)  # features per megapixel

        blur_scores.append(blur)
        feature_counts.append(feat_count)
        features_per_mp.append(feat_per_mp)

        aspect_ratio = float(w) / float(h)
        diagnostics["aspect_ratios"].append(aspect_ratio)

        # Conservative, deterministic filtering heuristic
        filter_flag = bool((blur < 50.0) or (feat_count < 200))
        diagnostics["filter_flags"].append(filter_flag)

        diagnostics["resolutions"].append({"width": int(w), "height": int(h)})
        diagnostics["blur_scores"].append(float(blur))
        diagnostics["feature_counts"].append(int(feat_count))
        diagnostics["features_per_megapixel"].append(feat_per_mp)

        if detailed_json:
            diagnostics["detailed_per_image"].append({
                "file": img_path.name,
                "width": w,
                "height": h,
                "aspect_ratio": aspect_ratio,
                "blur_score": blur,
                "feature_count": feat_count,
                "features_per_megapixel": feat_per_mp,
                "filter_flag": filter_flag
            })

    # General statistics
    diagnostics.update({
        "avg_blur": float(np.mean(blur_scores)) if blur_scores else 0.0,
        "min_blur": float(np.min(blur_scores)) if blur_scores else 0.0,
        "max_blur": float(np.max(blur_scores)) if blur_scores else 0.0,
        "avg_features": int(np.mean(feature_counts)) if feature_counts else 0,
        "min_features": int(np.min(feature_counts)) if feature_counts else 0,
        "max_features": int(np.max(feature_counts)) if feature_counts else 0,
        "avg_features_per_megapixel": float(np.mean(features_per_mp)) if features_per_mp else 0.0,
        "min_features_per_megapixel": float(np.min(features_per_mp)) if features_per_mp else 0.0,
        "max_features_per_megapixel": float(np.max(features_per_mp)) if features_per_mp else 0.0,
        "avg_aspect_ratio": float(np.mean(diagnostics["aspect_ratios"])) if diagnostics["aspect_ratios"] else 0.0
    })

    # Downsampling recommendation
    diagnostics["preprocessing"]["downsample_factor"] = (
        suggest_downsample(widths, heights) if widths and heights else 1.0
    )

    # Heuristic warnings
    if diagnostics["avg_blur"] < 100.0:
        diagnostics["recommendations"].append(
            "Average blur is low; aggressive filtering may be required"
        )
        logger.warning("Low average blur detected")

    if diagnostics["avg_features"] < 1000:
        diagnostics["recommendations"].append(
            "Low feature density; reconstruction robustness may suffer"
        )
        logger.warning("Low feature density detected")

    if diagnostics["image_count"] < 20:
        diagnostics["recommendations"].append(
            "Very few images; reconstruction may be incomplete"
        )
        logger.warning("Low image count detected")

    if diagnostics["avg_features_per_megapixel"] < 500:
        diagnostics["recommendations"].append(
            "Low features per megapixel; images may lack texture richness"
        )
        logger.warning("Low feature density per megapixel detected")

    # Coverage suggestion (simple heuristic)
    if diagnostics["image_count"] < 50:
        diagnostics["recommendations"].append(
            f"Only {diagnostics['image_count']} images; consider capturing more angles for complete mesh"
        )

    # --------------------------------------------------
    # Persist diagnostics
    # --------------------------------------------------

    diagnostics_path = paths.evaluation / "dataset_diagnostics.json"
    with open(diagnostics_path, "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)

    logger.info(f"Saved diagnostics JSON: {diagnostics_path}")

    # --------------------------------------------------
    # Plots
    # --------------------------------------------------

    try:
        # Resolution histogram
        plt.figure(figsize=(8, 4))
        plt.hist(widths, bins=10, alpha=0.7, label="Width")
        plt.hist(heights, bins=10, alpha=0.7, label="Height")
        plt.xlabel("Pixels")
        plt.ylabel("Frequency")
        plt.title("Resolution Distribution (raw)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(paths.evaluation / "resolution_distribution.png")
        plt.close()

        # Blur histogram
        plt.figure(figsize=(8, 4))
        plt.hist(blur_scores, bins=20)
        plt.xlabel("Blur score")
        plt.ylabel("Frequency")
        plt.title("Blur Score Distribution (raw)")
        plt.tight_layout()
        plt.savefig(paths.evaluation / "blur_score_distribution.png")
        plt.close()

        # Feature histogram
        plt.figure(figsize=(8, 4))
        plt.hist(feature_counts, bins=20)
        plt.xlabel("Feature count")
        plt.ylabel("Frequency")
        plt.title("Feature Density Distribution (raw)")
        plt.tight_layout()
        plt.savefig(paths.evaluation / "feature_density_distribution.png")
        plt.close()

        # Blur vs Features scatter
        plt.figure(figsize=(6, 6))
        plt.scatter(blur_scores, feature_counts, alpha=0.7)
        plt.xlabel("Blur score")
        plt.ylabel("Feature count")
        plt.title("Blur vs Feature Count (raw)")
        plt.tight_layout()
        plt.savefig(paths.evaluation / "blur_vs_features.png")
        plt.close()

        # Aspect ratio histogram
        plt.figure(figsize=(6, 4))
        plt.hist(diagnostics["aspect_ratios"], bins=15)
        plt.xlabel("Aspect ratio (w/h)")
        plt.ylabel("Frequency")
        plt.title("Aspect Ratio Distribution")
        plt.tight_layout()
        plt.savefig(paths.evaluation / "aspect_ratio_distribution.png")
        plt.close()

        logger.info("Saved diagnostic plots to evaluation/")
    except Exception as e:
        logger.warning(f"Plot generation failed: {e}")

    logger.info("Dataset analysis complete")
    return diagnostics

# --------------------------------------------------
# CLI wrapper
# --------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="MARK-2 Dataset Analyzer (raw)")
    parser.add_argument("--project", required=True, help="Project root directory")
    parser.add_argument("--force", action="store_true", help="Force re-analysis")
    parser.add_argument("--no-json-detail", action="store_true", help="Disable detailed per-image JSON output")
    args = parser.parse_args()

    run(Path(args.project), args.force, detailed_json=not args.no_json_detail)

if __name__ == "__main__":
    main()
