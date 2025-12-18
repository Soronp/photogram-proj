#!/usr/bin/env python3
"""
image_filter.py (MARK-2 SfM-SAFE)

Responsibilities:
- Drop only images that actively damage sparse reconstruction
- Never reason about overlap, angles, or perceptual similarity
- Never reduce dataset size meaningfully
- Guarantee geometric continuity
"""

import shutil
import json
import cv2
from pathlib import Path
from tqdm import tqdm

from utils.logger import get_logger
from utils.paths import ProjectPaths

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


# -------------------- METRICS --------------------
def keypoint_count(img):
    """
    Feature existence sanity check.
    ORB is fast and sufficient for rejection only.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=500)
    kp = orb.detect(gray, None)
    return len(kp)


# -------------------- PIPELINE --------------------
def run(project_root: Path, force=False):
    paths = ProjectPaths(project_root)
    paths.ensure_all()
    logger = get_logger("image_filter", project_root)

    src = paths.raw
    dst = paths.images_filtered

    if not src.exists():
        raise RuntimeError("raw/ missing")

    if dst.exists() and any(dst.iterdir()) and not force:
        logger.info("images_filtered exists — skipping")
        return

    if force and dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in src.iterdir() if p.suffix.lower() in SUPPORTED_EXTS)
    total = len(images)

    logger.info(f"Filtering {total} images (SfM-safe mode)")

    MIN_KEYPOINTS = 40        # catastrophic failure threshold
    MAX_DROP_RATIO = 0.10     # never drop more than 10%

    kept = []
    dropped = []

    for path in tqdm(images, desc="Filtering", unit="img"):
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

    # ---------------- SAFETY NET ----------------
    drop_ratio = len(dropped) / total if total else 0.0

    if drop_ratio > MAX_DROP_RATIO or len(kept) < max(8, int(total * 0.9)):
        logger.warning(
            "Filtering too destructive — reverting to no-op copy"
        )
        shutil.rmtree(dst)
        dst.mkdir(parents=True, exist_ok=True)

        kept = []
        dropped = []

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

    with open(dst / "filter_report.json", "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Kept {len(kept)} / Dropped {len(dropped)}")
