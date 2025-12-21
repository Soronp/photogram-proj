#!/usr/bin/env python3
"""
image_filter.py

MARK-2 SfM-SAFE Image Filtering Stage
------------------------------------
- Drops only catastrophic images
- Never reasons about overlap or similarity
- Guarantees geometric continuity
"""

import shutil
import json
import cv2
from pathlib import Path

from utils.paths import ProjectPaths

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


# --------------------------------------------------
# METRICS
# --------------------------------------------------

def keypoint_count(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=500)
    return len(orb.detect(gray, None))


# --------------------------------------------------
# PIPELINE STAGE
# --------------------------------------------------

def run(run_root: Path, project_root: Path, force: bool, logger):
    paths = ProjectPaths(project_root)
    paths.ensure_all()

    src = paths.raw
    dst = paths.images_filtered

    logger.info("[filter] Stage started (SfM-safe)")

    if not src.exists():
        raise RuntimeError("raw/ directory missing")

    if dst.exists() and any(dst.iterdir()) and not force:
        logger.info("[filter] images_filtered exists — skipping")
        return

    if dst.exists() and force:
        shutil.rmtree(dst)

    dst.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in src.iterdir() if p.suffix.lower() in SUPPORTED_EXTS)
    total = len(images)

    logger.info(f"[filter] Processing {total} images")

    MIN_KEYPOINTS = 40
    MAX_DROP_RATIO = 0.10

    kept = []
    dropped = []

    for path in images:
        img = cv2.imread(str(path))
        if img is None:
            dropped.append((path.name, "unreadable"))
            continue

        kps = keypoint_count(img)
        if kps < MIN_KEYPOINTS:
            dropped.append((path.name, f"no_features({kps})"))
            continue

        out_name = f"img_{len(kept):06d}.jpg"
        shutil.copy2(path, dst / out_name)
        kept.append(out_name)

    drop_ratio = len(dropped) / total if total else 0.0

    # --------------------------------------------------
    # SAFETY NET
    # --------------------------------------------------

    if drop_ratio > MAX_DROP_RATIO or len(kept) < max(8, int(total * 0.9)):
        logger.warning("[filter] Filtering too destructive — reverting to no-op copy")

        shutil.rmtree(dst)
        dst.mkdir(parents=True, exist_ok=True)

        kept.clear()
        dropped.clear()

        for i, path in enumerate(images):
            out = f"img_{i:06d}.jpg"
            shutil.copy2(path, dst / out)
            kept.append(out)

    report = {
        "input": total,
        "kept": len(kept),
        "dropped": len(dropped),
        "mode": "sfm-safe",
        "min_keypoints": MIN_KEYPOINTS,
    }

    with open(dst / "filter_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"[filter] Kept {len(kept)} / Dropped {len(dropped)}")
    logger.info("[filter] Stage completed")
