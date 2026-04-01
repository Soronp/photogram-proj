from pathlib import Path
import numpy as np
import json
import struct


# =====================================================
# PARAMS (KEEP YOUR DENSIFICATION STRATEGY)
# =====================================================
def _build_params():
    return {
        "max_image_size": -1,
        "min_num_pixels": 2,
        "max_num_pixels": 1000000,
        "max_traversal_depth": 10000,
        "cache_size": 4096,
        "use_cache": 1,
        "check_num_images": 2,
        "max_reproj_error": 1.5,
        "max_depth_error": 0.03,
        "max_normal_error": 30,
    }


# =====================================================
# COMMAND
# =====================================================
def _build_cmd(dense_dir, out_path, p):
    cmd = [
        "colmap", "stereo_fusion",
        "--workspace_path", str(dense_dir),
        "--workspace_format", "COLMAP",
        "--input_type", "geometric",
        "--output_path", str(out_path),
    ]

    for k, v in p.items():
        cmd.append(f"--StereoFusion.{k}")
        cmd.append(str(v))

    return cmd


# =====================================================
# FAST PLY READER (NO OPEN3D)
# =====================================================
def _read_ply_xyz(path):
    """
    Reads only XYZ from PLY (ASCII or binary little endian)
    Returns: Nx3 numpy array
    """
    with open(path, "rb") as f:
        header = []
        while True:
            line = f.readline().decode("utf-8").strip()
            header.append(line)
            if line == "end_header":
                break

        is_binary = any("binary_little_endian" in h for h in header)

        # Get vertex count
        vertex_count = 0
        for h in header:
            if h.startswith("element vertex"):
                vertex_count = int(h.split()[-1])

        if vertex_count == 0:
            return np.empty((0, 3))

        if not is_binary:
            # ASCII
            data = []
            for _ in range(vertex_count):
                vals = f.readline().decode("utf-8").split()
                data.append([float(vals[0]), float(vals[1]), float(vals[2])])
            return np.array(data)

        else:
            # Binary little endian
            pts = []
            for _ in range(vertex_count):
                x, y, z = struct.unpack("<fff", f.read(12))
                pts.append([x, y, z])
            return np.array(pts)


# =====================================================
# VALIDATION
# =====================================================
def _validate_points(points, logger):
    if len(points) == 0:
        raise RuntimeError("❌ No points in fused cloud")

    # Remove NaN / inf safely
    mask = np.isfinite(points).all(axis=1)
    clean = points[mask]

    removed = len(points) - len(clean)
    if removed > 0:
        logger.warning(f"[fusion] Removed {removed} invalid points")

    if len(clean) < 1000:
        logger.warning("[fusion] Very low point count")

    # Compute scale
    center = clean.mean(axis=0)
    dists = np.linalg.norm(clean - center, axis=1)
    scale = np.percentile(dists, 90)

    spread = np.linalg.norm(clean.max(axis=0) - clean.min(axis=0))
    logger.info(f"[fusion] Spread ratio: {spread / (scale + 1e-6):.2f}")

    return clean, scale


# =====================================================
# OPTIONAL LIGHT FILTER (SAFE)
# =====================================================
def _light_filter(points, logger):
    """
    Removes extreme outliers ONLY.
    Does NOT damage geometry.
    """
    if len(points) < 5000:
        return points

    center = points.mean(axis=0)
    dists = np.linalg.norm(points - center, axis=1)

    thresh = np.percentile(dists, 99.5)  # very loose
    mask = dists < thresh

    filtered = points[mask]

    logger.info(f"[fusion] Light filter: {len(points)} → {len(filtered)}")

    return filtered


# =====================================================
# METADATA
# =====================================================
def _save_metadata(points, scale, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "num_points": int(len(points)),
        "scale": float(scale),
    }

    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)


# =====================================================
# MAIN
# =====================================================
def run(paths, config, logger, tool_runner):
    logger.info("==== STEREO FUSION (COLMAP-ONLY STABLE) ====")

    dense_dir = paths.dense
    out_path = dense_dir / "fused.ply"
    meta_path = dense_dir / "fusion_metadata.json"

    # ---------------------------
    # RUN COLMAP
    # ---------------------------
    ret = tool_runner.run(
        _build_cmd(dense_dir, out_path, _build_params()),
        stage="fusion"
    )

    # 🔥 CRITICAL FIX: correct success check
    if ret["returncode"] != 0:
        raise RuntimeError(f"COLMAP stereo_fusion failed: {ret}")

    if not out_path.exists():
        raise RuntimeError("fused.ply not created")

    logger.info("[fusion] COLMAP finished successfully")

    # ---------------------------
    # READ POINTS (SAFE)
    # ---------------------------
    points = _read_ply_xyz(out_path)

    logger.info(f"[fusion] Raw points: {len(points)}")

    # ---------------------------
    # VALIDATION
    # ---------------------------
    points, scale = _validate_points(points, logger)

    # ---------------------------
    # OPTIONAL FILTER (SAFE)
    # ---------------------------
    points = _light_filter(points, logger)

    # ---------------------------
    # SAVE METADATA ONLY
    # (DO NOT rewrite PLY)
    # ---------------------------
    _save_metadata(points, scale, meta_path)

    logger.info(f"[fusion] Final points: {len(points)}")

    return {
        "status": "complete",
        "points": int(len(points)),
        "scale": float(scale)
    }