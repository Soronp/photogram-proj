import sqlite3
import numpy as np


def run(paths, config, logger):
    stage = "match_stats"
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
        raise RuntimeError(f"{stage}: no images found in database")

    # -----------------------------
    # Get matches (image pairs)
    # -----------------------------
    cursor.execute("SELECT pair_id FROM matches")
    pair_ids = [row[0] for row in cursor.fetchall()]

    num_matches = len(pair_ids)

    # -----------------------------
    # Decode pair_id → (i, j)
    # -----------------------------
    def pair_id_to_image_ids(pair_id):
        image_id2 = pair_id % 2147483647
        image_id1 = (pair_id - image_id2) // 2147483647
        return int(image_id1), int(image_id2)

    # -----------------------------
    # Build graph
    # -----------------------------
    degrees = [0] * (num_images + 1)  # 1-indexed

    for pid in pair_ids:
        i, j = pair_id_to_image_ids(pid)
        degrees[i] += 1
        degrees[j] += 1

    degrees = degrees[1:]  # remove index 0

    avg_degree = np.mean(degrees) if degrees else 0
    min_degree = np.min(degrees) if degrees else 0
    max_degree = np.max(degrees) if degrees else 0

    # -----------------------------
    # Connectivity
    # -----------------------------
    max_edges = num_images * (num_images - 1) / 2
    connectivity = num_matches / max_edges if max_edges > 0 else 0

    # -----------------------------
    # Weak node detection
    # -----------------------------
    weak_nodes = sum(1 for d in degrees if d < 2)

    stats = {
        "num_images": num_images,
        "num_matches": num_matches,
        "connectivity": float(connectivity),
        "avg_degree": float(avg_degree),
        "min_degree": int(min_degree),
        "max_degree": int(max_degree),
        "weak_nodes": int(weak_nodes),
    }

    logger.info(f"{stage}: stats computed")
    logger.info(f"{stage}: {stats}")

    conn.close()
    return stats