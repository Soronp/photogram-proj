#!/usr/bin/env python3
"""
image_filter.py

Pre-SfM image filtering stage (MARK-2 compliant).

Stage position:
    input  -> image_filter -> pre_processing

Responsibilities:
- Remove near-duplicate images (content-based)
- Detect and drop blurry / low-information images
- Preserve deterministic ordering
- Produce diagnostics for coverage and filtering

Reads:
- paths.images_processed

Writes:
- paths.images_filtered
"""

import shutil
import json
import cv2
import numpy as np
import hashlib
from pathlib import Path

from utils.logger import get_logger
from utils.paths import ProjectPaths
from utils.config import load_config


# -------------------------------------------------
# Image metrics
# -------------------------------------------------

def image_hash(img: np.ndarray) -> str:
    """Perceptual hash using downsampled grayscale image."""
    small = cv2.resize(img, (32, 32))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return hashlib.sha1(gray.tobytes()).hexdigest()


def blur_score(img: np.ndarray) -> float:
    """Variance of Laplacian (higher = sharper)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


# -------------------------------------------------
# Pipeline entrypoint
# -------------------------------------------------

def run(project_root: Path, force: bool):
    paths = ProjectPaths(project_root)
    paths.ensure_all()

    logger = get_logger("image_filter", project_root)
    config = load_config(project_root)

    src_dir = paths.images
    out_dir = paths.images_filtered

    if not src_dir.exists():
        raise RuntimeError("images_processed does not exist — input stage incomplete")

    # Skip logic
    if out_dir.exists() and any(out_dir.iterdir()) and not force:
        logger.info("images_filtered already exists; skipping")
        return

    # Force clean
    if force and out_dir.exists():
        logger.info("Force enabled — clearing images_filtered")
        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    blur_thresh = config.get("image_filter", {}).get("blur_threshold", 0.0)
    logger.info(f"Blur threshold: {blur_thresh}")

    seen_hashes = {}
    kept = []
    dropped = []

    frame_idx = 0
    images = sorted(src_dir.glob("*.jpg"))

    logger.info(f"Filtering {len(images)} images")

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            dropped.append({"file": img_path.name, "reason": "unreadable"})
            continue

        bscore = blur_score(img)
        if bscore < blur_thresh:
            dropped.append({
                "file": img_path.name,
                "reason": "blur",
                "score": float(bscore),
            })
            continue

        h = image_hash(img)
        if h in seen_hashes:
            dropped.append({
                "file": img_path.name,
                "reason": "duplicate",
                "duplicate_of": seen_hashes[h],
            })
            continue

        dst_name = f"img_{frame_idx:06d}.jpg"
        shutil.copy2(img_path, out_dir / dst_name)

        seen_hashes[h] = dst_name
        kept.append({
            "source": img_path.name,
            "output": dst_name,
            "blur_score": float(bscore),
        })

        frame_idx += 1

    report = {
        "input_count": len(images),
        "kept_count": len(kept),
        "dropped_count": len(dropped),
        "blur_threshold": blur_thresh,
        "kept": kept,
        "dropped": dropped,
    }

    report_path = out_dir / "filter_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(
        f"Filtering complete: kept {len(kept)} / dropped {len(dropped)}"
    )
