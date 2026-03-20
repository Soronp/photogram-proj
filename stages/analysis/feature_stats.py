import sqlite3
import numpy as np
import cv2
from pathlib import Path

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def run(paths, config, logger):
    stage = "feature_stats"
    logger.info(f"---- {stage.upper()} ----")

    db_path = paths.database

    if not db_path.exists():
        raise RuntimeError(f"{stage}: database.db not found")

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    try:
        # -----------------------------
        # Image count
        # -----------------------------
        cursor.execute("SELECT COUNT(*) FROM images")
        num_images = cursor.fetchone()[0]

        if num_images == 0:
            raise RuntimeError(f"{stage}: no images in database")

        # -----------------------------
        # Keypoints per image
        # -----------------------------
        cursor.execute("SELECT rows FROM keypoints")
        keypoints_counts = [row[0] for row in cursor.fetchall()]

        if len(keypoints_counts) == 0:
            logger.warning(f"{stage}: no keypoints found")
            avg_keypoints = 0
            total_keypoints = 0
        else:
            total_keypoints = int(sum(keypoints_counts))
            avg_keypoints = float(np.mean(keypoints_counts))

        # -----------------------------
        # Image area estimation (ROBUST)
        # -----------------------------
        image_dir = paths.images

        image_files = [
            p for p in image_dir.iterdir()
            if p.suffix in VALID_EXTENSIONS
        ]

        areas = []

        for img_path in image_files[:10]:  # sample first 10
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            areas.append(h * w)

        if len(areas) == 0:
            logger.warning(f"{stage}: could not compute image areas")
            avg_area = 1  # prevent division by zero
        else:
            avg_area = float(np.mean(areas))

        feature_density = avg_keypoints / avg_area

        # -----------------------------
        # Stats
        # -----------------------------
        stats = {
            "num_images": int(num_images),
            "total_keypoints": total_keypoints,
            "avg_keypoints_per_image": avg_keypoints,
            "feature_density": float(feature_density),
        }

        logger.info(f"{stage}: SUCCESS")
        logger.info(f"{stage}: {stats}")

        return stats

    finally:
        conn.close()