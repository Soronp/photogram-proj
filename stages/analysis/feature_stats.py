import sqlite3
import numpy as np


def run(paths, config, logger):
    stage = "feature_stats"
    logger.info(f"---- {stage.upper()} ----")

    db_path = paths.database

    if not db_path.exists():
        raise RuntimeError(f"{stage}: database.db not found")

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # -----------------------------
    # Get number of images
    # -----------------------------
    cursor.execute("SELECT COUNT(*) FROM images")
    num_images = cursor.fetchone()[0]

    if num_images == 0:
        raise RuntimeError(f"{stage}: no images in database")

    # -----------------------------
    # Keypoints
    # -----------------------------
    cursor.execute("SELECT rows FROM keypoints")
    keypoints_counts = [row[0] for row in cursor.fetchall()]

    total_keypoints = sum(keypoints_counts)
    avg_keypoints = np.mean(keypoints_counts) if keypoints_counts else 0

    # -----------------------------
    # Matches
    # -----------------------------
    cursor.execute("SELECT COUNT(*) FROM matches")
    num_matches = cursor.fetchone()[0]

    # -----------------------------
    # Connectivity
    # -----------------------------
    max_possible_edges = num_images * (num_images - 1) / 2

    connectivity = (
        num_matches / max_possible_edges if max_possible_edges > 0 else 0
    )

    # -----------------------------
    # Feature density
    # -----------------------------
    downsample_enabled = config.get("downsampling", {}).get("enabled", True)

    if downsample_enabled:
        image_dir = paths.images_downsampled
    else:
        image_dir = paths.images

    # Estimate image area (use first image)
    import cv2
    sample_img = next(image_dir.glob("*"))
    img = cv2.imread(str(sample_img))
    h, w = img.shape[:2]

    image_area = h * w

    feature_density = avg_keypoints / image_area if image_area > 0 else 0

    # -----------------------------
    # Output stats
    # -----------------------------
    stats = {
        "num_images": num_images,
        "total_keypoints": int(total_keypoints),
        "avg_keypoints_per_image": float(avg_keypoints),
        "num_matches": int(num_matches),
        "connectivity": float(connectivity),
        "feature_density": float(feature_density),
    }

    logger.info(f"{stage}: stats computed")
    logger.info(f"{stage}: {stats}")

    conn.close()

    return stats