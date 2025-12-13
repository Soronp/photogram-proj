#!/usr/bin/env python3
"""
Mark-2 Image Preprocessing Module (2K-safe)
- Preserves aspect ratio
- Scales images ONLY if max dimension exceeds 2000 px
- Avoids geometric distortion (COLMAP-safe)
- JSON-safe metadata
"""

import os
import cv2
import json
import time
from utils.config import PATHS
from utils.logger import get_logger

logger = get_logger()

METADATA_FILE = os.path.join(PATHS["processed"], "preprocessing_metadata.json")
MAX_SIDE = 2000  # hard upper bound, NOT forced resolution


# ----------------------------------------------------------------------
# Single-image preprocessing
# ----------------------------------------------------------------------
def preprocess_image(img_path, output_folder, adjust_contrast=True):
    os.makedirs(output_folder, exist_ok=True)

    img = cv2.imread(img_path)
    if img is None:
        logger.warning(f"Failed to read image: {img_path}")
        return None

    h, w = img.shape[:2]
    max_dim = max(w, h)

    # --- Scale ONLY if image exceeds MAX_SIDE ---
    scale_factor = 1.0
    if max_dim > MAX_SIDE:
        scale_factor = MAX_SIDE / max_dim
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        new_w, new_h = w, h

    # --- Optional mild contrast enhancement (COLMAP-safe) ---
    alpha, beta = 1.0, 0
    if adjust_contrast:
        alpha = 1.1  # conservative contrast
        beta = 5     # minimal brightness shift
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    # Save image
    filename = os.path.basename(img_path)
    save_path = os.path.join(output_folder, filename)
    cv2.imwrite(save_path, img)

    logger.info(
        f"Preprocessed {filename}: "
        f"{w}x{h} → {new_w}x{new_h} (scale={round(scale_factor,3)})"
    )

    return {
        "output_path": str(save_path),
        "original_size": [w, h],
        "processed_size": [new_w, new_h],
        "scale_factor": scale_factor,
        "contrast_alpha": alpha,
        "brightness_beta": beta,
    }


# ----------------------------------------------------------------------
# Folder preprocessing
# ----------------------------------------------------------------------
def run_preprocessing(input_folder=None, output_folder=None, adjust_contrast=True):
    start_time = time.time()

    if input_folder is None:
        input_folder = PATHS["filtered"]
    if output_folder is None:
        output_folder = os.path.join(PATHS["processed"], "images_preprocessed")

    os.makedirs(output_folder, exist_ok=True)

    images = [
        f for f in os.listdir(input_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff"))
    ]

    logger.info(f"Found {len(images)} images in {input_folder}")

    metadata = {}
    ok, failed = 0, 0

    for name in images:
        path = os.path.join(input_folder, name)
        try:
            result = preprocess_image(path, output_folder, adjust_contrast)
            if result:
                metadata[name] = result
                ok += 1
            else:
                failed += 1
        except Exception as e:
            logger.error(f"Failed {name}: {e}")
            failed += 1

    os.makedirs(PATHS["processed"], exist_ok=True)
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Metadata saved to {METADATA_FILE}")
    logger.info(
        f"Preprocessing complete: {ok}/{len(images)} OK, "
        f"{failed} failed in {round(time.time() - start_time, 2)}s"
    )

    return str(output_folder)


# ----------------------------------------------------------------------
# Standalone execution
# ----------------------------------------------------------------------
if __name__ == "__main__":
    input_dir = input("Enter input folder:\n> ").strip('" ')
    output_dir = input("Enter output folder:\n> ").strip('" ')
    run_preprocessing(input_dir, output_dir, adjust_contrast=True)
