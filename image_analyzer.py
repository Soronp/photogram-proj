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

def run(project_root: Path, force: bool = False):
    paths = ProjectPaths(project_root)
    paths.ensure_all()

    logger = get_logger("image_analyzer", project_root)
    logger.info("Starting dataset analysis (raw images)")

    raw_dir = paths.raw
    images = [
        p for p in sorted(raw_dir.iterdir())
        if p.suffix.lower() in SUPPORTED_IMAGE_EXTS
    ]

    if not images:
        logger.warning("No images found in raw/ for analysis")
        return None

    diagnostics = {
        "image_count": int(len(images)),
        "resolutions": [],
        "blur_scores": [],
        "feature_counts": [],
        "filter_flags": [],
        "preprocessing": {},
        "recommendations": [],
    }

    widths, heights = [], []
    blur_scores, feature_counts = [], []

    for img_path in tqdm(images, desc="Analyzing raw images"):
        img = cv2.imread(str(img_path))

        if img is None:
            logger.warning(f"Unreadable image: {img_path.name}")
            diagnostics["filter_flags"].append(True)
            continue

        h, w = img.shape[:2]
        widths.append(int(w))
        heights.append(int(h))

        blur = compute_blur_score(img)
        feat_count = compute_feature_density(img)

        blur_scores.append(blur)
        feature_counts.append(feat_count)

        # Conservative, deterministic filtering heuristic
        filter_flag = bool((blur < 50.0) or (feat_count < 200))

        diagnostics["filter_flags"].append(filter_flag)
        diagnostics["resolutions"].append({"width": int(w), "height": int(h)})
        diagnostics["blur_scores"].append(float(blur))
        diagnostics["feature_counts"].append(int(feat_count))

    diagnostics["avg_blur"] = float(np.mean(blur_scores)) if blur_scores else 0.0
    diagnostics["avg_features"] = int(np.mean(feature_counts)) if feature_counts else 0

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

        plt.figure(figsize=(8, 4))
        plt.hist(blur_scores, bins=20)
        plt.xlabel("Blur score")
        plt.ylabel("Frequency")
        plt.title("Blur Score Distribution (raw)")
        plt.tight_layout()
        plt.savefig(paths.evaluation / "blur_score_distribution.png")
        plt.close()

        plt.figure(figsize=(8, 4))
        plt.hist(feature_counts, bins=20)
        plt.xlabel("Feature count")
        plt.ylabel("Frequency")
        plt.title("Feature Density Distribution (raw)")
        plt.tight_layout()
        plt.savefig(paths.evaluation / "feature_density_distribution.png")
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
    args = parser.parse_args()

    run(Path(args.project), args.force)

if __name__ == "__main__":
    main()
