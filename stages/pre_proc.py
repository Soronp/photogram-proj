#!/usr/bin/env python3
"""
Stage 2 — Image preprocessing

Normalizes images for consistent SfM behaviour.
"""

from pathlib import Path
import shutil
import cv2


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def is_image(p: Path):
    return p.suffix.lower() in SUPPORTED_EXTS


def run(paths, logger, tools, config):

    logger.info("[pre_proc] stage started")

    src = paths.images
    dst = paths.images_preprocessed

    if not src.exists():
        raise RuntimeError("input images directory missing")

    if dst.exists():
        shutil.rmtree(dst)

    dst.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in src.iterdir() if is_image(p))

    if not images:
        raise RuntimeError("No images found for preprocessing")

    logger.info(f"[pre_proc] processing {len(images)} images")

    processed = 0

    for i, path in enumerate(images):

        img = cv2.imread(str(path))

        if img is None:
            logger.warning(f"[pre_proc] unreadable: {path.name}")
            continue

        output_name = f"img_{processed:06d}.jpg"

        out_path = dst / output_name

        cv2.imwrite(str(out_path), img)

        processed += 1

    logger.info(f"[pre_proc] wrote {processed} images")

    if processed == 0:
        raise RuntimeError("Preprocessing produced zero images")

    logger.info("[pre_proc] stage completed")