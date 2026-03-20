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

    try:
        # -----------------------------
        # IMAGE COUNT
        # -----------------------------
        cursor.execute("SELECT COUNT(*) FROM images")
        num_images = cursor.fetchone()[0]

        if num_images == 0:
            raise RuntimeError(f"{stage}: no images in database")

        # -----------------------------
        # MATCH DATA
        # -----------------------------
        cursor.execute("SELECT pair_id, rows FROM matches")
        rows = cursor.fetchall()

        if not rows:
            logger.warning(f"{stage}: no matches found")

            return {
                "num_images": num_images,
                "num_matches": 0,
                "avg_matches_per_pair": 0.0,
                "connectivity": 0.0,
                "avg_degree": 0.0,
                "min_degree": 0,
                "max_degree": 0,
                "weak_nodes": num_images,
            }

        # -----------------------------
        # DECODE GRAPH
        # -----------------------------
        MAX_ID = 2147483647

        def decode_pair(pair_id):
            j = pair_id % MAX_ID
            i = (pair_id - j) // MAX_ID
            return int(i), int(j)

        degrees = {}

        valid_pairs = 0
        match_sizes = []

        for pair_id, rows_count in rows:
            i, j = decode_pair(pair_id)

            # Skip invalid IDs
            if i <= 0 or j <= 0:
                continue

            degrees[i] = degrees.get(i, 0) + 1
            degrees[j] = degrees.get(j, 0) + 1

            valid_pairs += 1
            match_sizes.append(rows_count)

        if valid_pairs == 0:
            logger.warning(f"{stage}: no valid match pairs")

            return {
                "num_images": num_images,
                "num_matches": 0,
                "avg_matches_per_pair": 0.0,
                "connectivity": 0.0,
                "avg_degree": 0.0,
                "min_degree": 0,
                "max_degree": 0,
                "weak_nodes": num_images,
            }

        # -----------------------------
        # DEGREE VECTOR
        # -----------------------------
        degree_values = np.array(list(degrees.values()))

        avg_degree = float(np.mean(degree_values))
        min_degree = int(np.min(degree_values))
        max_degree = int(np.max(degree_values))

        weak_nodes = int(sum(1 for d in degree_values if d < 2))

        # -----------------------------
        # CONNECTIVITY
        # -----------------------------
        max_edges = num_images * (num_images - 1) / 2
        connectivity = valid_pairs / max_edges if max_edges > 0 else 0.0

        # -----------------------------
        # MATCH QUALITY
        # -----------------------------
        avg_matches_per_pair = float(np.mean(match_sizes)) if match_sizes else 0.0

        stats = {
            "num_images": num_images,
            "num_matches": valid_pairs,
            "avg_matches_per_pair": avg_matches_per_pair,
            "connectivity": float(connectivity),
            "avg_degree": avg_degree,
            "min_degree": min_degree,
            "max_degree": max_degree,
            "weak_nodes": weak_nodes,
        }

        logger.info(f"{stage}: SUCCESS")
        logger.info(f"{stage}: {stats}")

        return stats

    finally:
        conn.close()