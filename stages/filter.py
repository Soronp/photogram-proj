#!/usr/bin/env python3
"""
filter.py

Stage — SfM-safe image filtering

Drops only catastrophic images.
Never reasons about overlap.
"""

from pathlib import Path
import shutil
import json
import cv2


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

MIN_KEYPOINTS = 40
MAX_DROP_RATIO = 0.10


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def keypoint_count(img):
    """Compute ORB keypoint count."""

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=500)

    keypoints = orb.detect(gray, None)

    return len(keypoints)


def is_image(path: Path):
    return path.suffix.lower() in SUPPORTED_EXTS


# --------------------------------------------------
# Stage execution
# --------------------------------------------------

def run(paths, logger, tools, config):

    logger.info("[filter] stage started")

    src = paths.images_preprocessed
    dst = paths.images_filtered

    # --------------------------------------------------
    # Fallback if preprocessing was skipped
    # --------------------------------------------------

    if not src.exists() or not any(src.iterdir()):

        logger.warning(
            "[filter] images_preprocessed empty — falling back to ingestion images"
        )

        src = paths.images

    if not src.exists():
        raise RuntimeError("No source images available")

    if dst.exists():
        shutil.rmtree(dst)

    dst.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in src.iterdir() if is_image(p))

    total = len(images)

    if total == 0:
        raise RuntimeError("No images found for filtering")

    logger.info(f"[filter] processing {total} images")

    kept = []
    dropped = []

    # --------------------------------------------------
    # Filtering loop
    # --------------------------------------------------

    for path in images:

        img = cv2.imread(str(path))

        if img is None:
            dropped.append((path.name, "unreadable"))
            continue

        kps = keypoint_count(img)

        if kps < MIN_KEYPOINTS:
            dropped.append((path.name, f"low_features({kps})"))
            continue

        output_name = f"img_{len(kept):06d}.jpg"

        shutil.copy2(path, dst / output_name)

        kept.append(output_name)

    drop_ratio = len(dropped) / max(1, total)

    # --------------------------------------------------
    # Safety fallback
    # --------------------------------------------------

    if drop_ratio > MAX_DROP_RATIO or len(kept) < max(8, int(total * 0.9)):

        logger.warning("[filter] filtering too aggressive — reverting")

        shutil.rmtree(dst)
        dst.mkdir(parents=True, exist_ok=True)

        kept.clear()
        dropped.clear()

        for i, path in enumerate(images):

            output_name = f"img_{i:06d}.jpg"

            shutil.copy2(path, dst / output_name)

            kept.append(output_name)

    # --------------------------------------------------
    # Report
    # --------------------------------------------------

    report = {
        "input": total,
        "kept": len(kept),
        "dropped": len(dropped),
        "mode": "sfm-safe",
        "min_keypoints": MIN_KEYPOINTS
    }

    with open(dst / "filter_report.json", "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"[filter] kept {len(kept)} / dropped {len(dropped)}")
    logger.info("[filter] stage completed")