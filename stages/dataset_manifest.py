#!/usr/bin/env python3
"""
data_manifest.py

Creates a dataset manifest for reproducibility and diagnostics.

Outputs:
    dataset_manifest.json

Contents:
    • image list
    • SHA256 checksums
    • deterministic dataset hash
    • image metadata
    • basic quality diagnostics

Runs in background so pipeline execution is not blocked.
"""

import hashlib
import json
import threading
from pathlib import Path

import cv2
from PIL import Image, ExifTags
import numpy as np


IMAGE_EXT = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


# --------------------------------------------------
# File Hash
# --------------------------------------------------

def sha256_file(path: Path):

    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


# --------------------------------------------------
# Dataset Scanner
# --------------------------------------------------

def collect_images(dataset_path: Path):

    images = []

    for f in dataset_path.rglob("*"):

        if f.suffix.lower() in IMAGE_EXT:
            images.append(f)

    images.sort()

    return images


# --------------------------------------------------
# EXIF Extraction
# --------------------------------------------------

def extract_exif(image_path):

    exif_data = {}

    try:

        img = Image.open(image_path)

        exif = img._getexif()

        if exif:

            tag_map = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}

            exif_data["camera_make"] = tag_map.get("Make")
            exif_data["camera_model"] = tag_map.get("Model")
            exif_data["focal_length"] = str(tag_map.get("FocalLength"))
            exif_data["exposure_time"] = str(tag_map.get("ExposureTime"))
            exif_data["iso"] = tag_map.get("ISOSpeedRatings")

    except Exception:
        pass

    return exif_data


# --------------------------------------------------
# Image Diagnostics
# --------------------------------------------------

def compute_image_diagnostics(image_path):

    diagnostics = {}

    try:

        img = cv2.imread(str(image_path))

        if img is None:
            return diagnostics

        height, width = img.shape[:2]

        diagnostics["width"] = int(width)
        diagnostics["height"] = int(height)
        diagnostics["megapixels"] = round((width * height) / 1e6, 2)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Blur score (variance of Laplacian)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

        diagnostics["blur_score"] = round(float(blur_score), 2)

        # brightness
        diagnostics["brightness"] = round(float(np.mean(gray)), 2)

    except Exception:
        pass

    return diagnostics


# --------------------------------------------------
# Manifest Generator
# --------------------------------------------------

def _generate_manifest(paths, config, logger):

    dataset_path = Path(config["project"]["dataset_path"])

    manifest_path = paths.root / "dataset_manifest.json"

    if manifest_path.exists():

        logger.info("[data_manifest] manifest already exists")

        return

    logger.info("[data_manifest] hashing dataset in background")

    images = collect_images(dataset_path)

    image_entries = []

    for img in images:

        checksum = sha256_file(img)

        entry = {
            "file": img.name,
            "path": str(img),
            "sha256": checksum
        }

        # EXIF metadata
        entry.update(extract_exif(img))

        # image diagnostics
        entry.update(compute_image_diagnostics(img))

        image_entries.append(entry)

    # deterministic dataset hash

    dataset_hash = hashlib.sha256(

        "".join(i["sha256"] for i in image_entries).encode()

    ).hexdigest()

    manifest = {

        "dataset_path": str(dataset_path),

        "image_count": len(image_entries),

        "dataset_hash": dataset_hash,

        "images": image_entries
    }

    with open(manifest_path, "w", encoding="utf-8") as f:

        json.dump(manifest, f, indent=2)

    logger.info(
        f"[data_manifest] completed ({len(image_entries)} images)"
    )


# --------------------------------------------------
# Background Worker
# --------------------------------------------------

def start_manifest_worker(paths, config, logger):

    thread = threading.Thread(

        target=_generate_manifest,

        args=(paths, config, logger),

        daemon=True
    )

    thread.start()

    logger.info("[data_manifest] background job started")

    return thread


# --------------------------------------------------
# Pipeline Stage Entry
# --------------------------------------------------

def run(paths, tools, config, logger):

    logger.info("[data_manifest] stage start")

    start_manifest_worker(paths, config, logger)

    logger.info("[data_manifest] stage completed (background hashing)")