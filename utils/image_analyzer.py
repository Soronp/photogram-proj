#!/usr/bin/env python3

import json
import cv2
import numpy as np
from collections import Counter


SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def _collect_images(directory):

    if not directory.exists():
        return []

    return sorted(
        p for p in directory.iterdir()
        if p.suffix.lower() in SUPPORTED_IMAGE_EXTS
    )


def compute_blur(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def run(paths, logger, tools, config):

    filtered = _collect_images(paths.images_filtered)
    processed = _collect_images(paths.images_processed)

    if filtered:
        images = filtered
        source = "filtered"
    elif processed:
        images = processed
        source = "processed"
    else:
        raise RuntimeError("no images found")

    eval_dir = paths.evaluation
    eval_dir.mkdir(exist_ok=True)

    out = eval_dir / "dataset_intelligence.json"

    logger.info(f"[analyzer] {len(images)} images from {source}")

    blur_vals = []

    for img_path in images:

        img = cv2.imread(str(img_path))

        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        blur_vals.append(compute_blur(gray))

    result = {
        "image_count": len(images),
        "blur_mean": float(np.mean(blur_vals)),
        "blur_low_ratio": float(np.mean(np.array(blur_vals) < 100))
    }

    with open(out, "w") as f:
        json.dump(result, f, indent=2)

    logger.info("[analyzer] dataset intelligence written")