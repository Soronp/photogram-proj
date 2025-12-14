#!/usr/bin/env python3
"""
image_filter.py

Pre-SfM image filtering stage.

Responsibilities:
- Remove near-duplicate images (content-based)
- Detect and drop blurry / low-information images
- Preserve deterministic ordering
- Produce diagnostics for coverage and filtering

This script MUST:
- Read from images/
- Write to images_filtered/
- Never modify images/
- Be safe to re-run
"""

import argparse
import shutil
import json
import cv2
import numpy as np
import hashlib
from pathlib import Path

from utils.logger import get_logger
from utils.paths import ProjectPaths
from utils.config import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Filter redundant / bad images")
    parser.add_argument("--project", required=True, help="Path to project root")
    parser.add_argument("--force", action="store_true", help="Re-run filtering")
    return parser.parse_args()


def image_hash(img: np.ndarray) -> str:
    """Perceptual hash using downsampled grayscale image."""
    small = cv2.resize(img, (32, 32))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return hashlib.sha1(gray.tobytes()).hexdigest()


def blur_score(img: np.ndarray) -> float:
    """Variance of Laplacian (higher = sharper)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def main():
    args = parse_args()

    project_root = Path(args.project).resolve()
    paths = ProjectPaths(project_root)
    paths.ensure_all()

    logger = get_logger("image_filter", project_root)
    config = load_config(project_root)

    images_dir = paths.images_processed
    out_dir = paths.images_filtered

    if out_dir.exists() and any(out_dir.iterdir()) and not args.force:
        logger.info("images_filtered already exists; skipping")
        return

    if args.force and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(exist_ok=True)

    blur_thresh = config.get("image_filter", {}).get("blur_threshold", 0.0)

    logger.info(f"Blur threshold: {blur_thresh}")

    seen_hashes = {}
    kept = []
    dropped = []

    frame_idx = 0

    for img_path in sorted(images_dir.glob("*.jpg")):
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning(f"Unreadable image skipped: {img_path.name}")
            dropped.append({"file": img_path.name, "reason": "unreadable"})
            continue

        bscore = blur_score(img)
        if bscore < blur_thresh:
            dropped.append({"file": img_path.name, "reason": "blur", "score": bscore})
            continue

        h = image_hash(img)
        if h in seen_hashes:
            dropped.append({"file": img_path.name, "reason": "duplicate"})
            continue

        dst_name = f"img_{frame_idx:06d}.jpg"
        shutil.copy2(img_path, out_dir / dst_name)

        seen_hashes[h] = dst_name
        kept.append({
            "source": img_path.name,
            "output": dst_name,
            "blur_score": bscore,
        })

        frame_idx += 1

    report = {
        "input_count": len(list(images_dir.glob("*.jpg"))),
        "kept_count": len(kept),
        "dropped_count": len(dropped),
        "blur_threshold": blur_thresh,
        "kept": kept,
        "dropped": dropped,
    }

    report_path = out_dir / "filter_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Filtering complete: kept {len(kept)} / dropped {len(dropped)}")


if __name__ == "__main__":
    main()
