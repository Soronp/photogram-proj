from pathlib import Path
import numpy as np
import struct
import tempfile
import shutil
import json


# =====================================================
# FUSION PROFILES
# =====================================================
def _build_fusion_profiles():
    """
    Ordered strict → balanced → recovery
    Geometry stability prioritized over density.
    """
    return [
        {
            "name": "strict",
            "params": {
                "max_image_size": 2400,
                "min_num_pixels": 5,
                "max_num_pixels": 500000,
                "max_traversal_depth": 100,
                "check_num_images": 4,
                "max_reproj_error": 0.75,
                "max_depth_error": 0.008,
                "max_normal_error": 15,
                "cache_size": 4096,
                "use_cache": 1,
            }
        },
        {
            "name": "balanced",
            "params": {
                "max_image_size": 2600,
                "min_num_pixels": 4,
                "max_num_pixels": 750000,
                "max_traversal_depth": 150,
                "check_num_images": 3,
                "max_reproj_error": 1.0,
                "max_depth_error": 0.012,
                "max_normal_error": 20,
                "cache_size": 4096,
                "use_cache": 1,
            }
        },
        {
            "name": "recovery",
            "params": {
                "max_image_size": -1,
                "min_num_pixels": 3,
                "max_num_pixels": 1000000,
                "max_traversal_depth": 200,
                "check_num_images": 3,
                "max_reproj_error": 1.25,
                "max_depth_error": 0.018,
                "max_normal_error": 22,
                "cache_size": 4096,
                "use_cache": 1,
            }
        }
    ]


# =====================================================
# COMMAND BUILDER
# =====================================================
def _build_cmd(dense_dir, out_path, params):
    cmd = [
        "colmap", "stereo_fusion",
        "--workspace_path", str(dense_dir),
        "--workspace_format", "COLMAP",
        "--input_type", "geometric",
        "--output_path", str(out_path),
    ]

    for k, v in params.items():
        cmd += [f"--StereoFusion.{k}", str(v)]

    return cmd


# =====================================================
# TYPE MAP
# =====================================================
PLY_TYPES = {
    "char": ("b", 1),
    "uchar": ("B", 1),
    "int8": ("b", 1),
    "uint8": ("B", 1),
    "short": ("h", 2),
    "ushort": ("H", 2),
    "int16": ("h", 2),
    "uint16": ("H", 2),
    "int": ("i", 4),
    "uint": ("I", 4),
    "int32": ("i", 4),
    "uint32": ("I", 4),
    "float": ("f", 4),
    "float32": ("f", 4),
    "double": ("d", 8),
    "float64": ("d", 8),
}


# =====================================================
# FULL SAFE PLY READER
# Preserves:
# x y z nx ny nz r g b
# =====================================================
def _read_ply_full(path):
    with open(path, "rb") as f:
        header = []
        properties = []
        vertex_count = 0

        while True:
            line = f.readline().decode("utf-8").strip()
            header.append(line)

            if line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])

            elif line.startswith("property"):
                parts = line.split()

                # Ignore list properties
                if len(parts) == 3:
                    properties.append((parts[1], parts[2]))

            elif line == "end_header":
                break

        if vertex_count == 0:
            raise RuntimeError("PLY contains zero vertices")

        is_binary = any("binary_little_endian" in h for h in header)

        prop_names = [p[1] for p in properties]

        required_xyz = ["x", "y", "z"]
        for req in required_xyz:
            if req not in prop_names:
                raise RuntimeError(f"Missing required property: {req}")

        # Optional properties
        has_normals = all(p in prop_names for p in ("nx", "ny", "nz"))
        has_rgb = all(p in prop_names for p in ("red", "green", "blue"))

        xyz = np.zeros((vertex_count, 3), dtype=np.float32)
        normals = np.zeros((vertex_count, 3), dtype=np.float32)
        rgb = np.full((vertex_count, 3), 255, dtype=np.uint8)

        # =================================================
        # ASCII
        # =================================================
        if not is_binary:
            for i in range(vertex_count):
                vals = f.readline().decode("utf-8").split()

                xyz[i] = [
                    float(vals[prop_names.index("x")]),
                    float(vals[prop_names.index("y")]),
                    float(vals[prop_names.index("z")]),
                ]

                if has_normals:
                    normals[i] = [
                        float(vals[prop_names.index("nx")]),
                        float(vals[prop_names.index("ny")]),
                        float(vals[prop_names.index("nz")]),
                    ]

                if has_rgb:
                    rgb[i] = [
                        int(vals[prop_names.index("red")]),
                        int(vals[prop_names.index("green")]),
                        int(vals[prop_names.index("blue")]),
                    ]

            return xyz, normals, rgb

        # =================================================
        # BINARY
        # =================================================
        fmt = "<"
        for ptype, _ in properties:
            if ptype not in PLY_TYPES:
                raise RuntimeError(f"Unsupported PLY type: {ptype}")
            fmt += PLY_TYPES[ptype][0]

        vertex_size = struct.calcsize(fmt)

        x_idx = prop_names.index("x")
        y_idx = prop_names.index("y")
        z_idx = prop_names.index("z")

        nx_idx = prop_names.index("nx") if has_normals else None
        ny_idx = prop_names.index("ny") if has_normals else None
        nz_idx = prop_names.index("nz") if has_normals else None

        r_idx = prop_names.index("red") if has_rgb else None
        g_idx = prop_names.index("green") if has_rgb else None
        b_idx = prop_names.index("blue") if has_rgb else None

        for i in range(vertex_count):
            raw = f.read(vertex_size)

            if len(raw) != vertex_size:
                raise RuntimeError("PLY truncated during vertex read")

            vals = struct.unpack(fmt, raw)

            xyz[i] = [vals[x_idx], vals[y_idx], vals[z_idx]]

            if has_normals:
                normals[i] = [vals[nx_idx], vals[ny_idx], vals[nz_idx]]

            if has_rgb:
                rgb[i] = [vals[r_idx], vals[g_idx], vals[b_idx]]

        return xyz, normals, rgb


# =====================================================
# FULL SAFE PLY WRITER
# Downstream compatible:
# x y z nx ny nz r g b
# =====================================================
def _write_ply_full(path, xyz, normals, rgb):
    with open(path, "wb") as f:
        header = "\n".join([
            "ply",
            "format binary_little_endian 1.0",
            f"element vertex {len(xyz)}",
            "property float x",
            "property float y",
            "property float z",
            "property float nx",
            "property float ny",
            "property float nz",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "end_header\n"
        ])

        f.write(header.encode("utf-8"))

        for p, n, c in zip(xyz, normals, rgb):
            f.write(struct.pack(
                "<ffffffBBB",
                float(p[0]), float(p[1]), float(p[2]),
                float(n[0]), float(n[1]), float(n[2]),
                int(c[0]), int(c[1]), int(c[2]),
            ))


# =====================================================
# VALIDATION
# =====================================================
def _validate_cloud(xyz):
    finite_mask = np.isfinite(xyz).all(axis=1)
    xyz_valid = xyz[finite_mask]

    if len(xyz_valid) < 500:
        raise RuntimeError("Too few valid points")

    center = xyz_valid.mean(axis=0)
    dists = np.linalg.norm(xyz_valid - center, axis=1)

    scale = np.percentile(dists, 90)
    spread = np.linalg.norm(
        xyz_valid.max(axis=0) - xyz_valid.min(axis=0)
    )

    spread_ratio = spread / (scale + 1e-8)

    return finite_mask, scale, spread_ratio


# =====================================================
# LIGHT OUTLIER FILTER
# =====================================================
def _light_filter(xyz):
    if len(xyz) < 5000:
        return np.ones(len(xyz), dtype=bool)

    center = xyz.mean(axis=0)
    dists = np.linalg.norm(xyz - center, axis=1)

    threshold = np.percentile(dists, 99.5)

    return dists < threshold


# =====================================================
# SCORE
# =====================================================
def _score_cloud(num_points, spread_ratio):
    return num_points / max(spread_ratio, 1e-6)


# =====================================================
# METADATA
# =====================================================
def _save_metadata(path, profile, points, scale, spread):
    metadata = {
        "profile": profile,
        "points": int(points),
        "scale": float(scale),
        "spread_ratio": float(spread),
    }

    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)


# =====================================================
# MAIN
# =====================================================
def run(paths, config, logger, tool_runner):
    logger.info("==== ROBUST FULL-COMPAT STEREO FUSION ====")

    dense_dir = paths.dense
    final_out = dense_dir / "fused.ply"
    meta_out = dense_dir / "fusion_metadata.json"

    profiles = _build_fusion_profiles()

    best_score = -1
    best_xyz = None
    best_normals = None
    best_rgb = None
    best_profile = None
    best_scale = None
    best_spread = None

    tmp_dir = Path(tempfile.mkdtemp(prefix="fusion_trials_"))

    try:
        for profile in profiles:
            logger.info(f"[fusion] Running profile: {profile['name']}")

            trial_out = tmp_dir / f"{profile['name']}.ply"

            ret = tool_runner.run(
                _build_cmd(dense_dir, trial_out, profile["params"]),
                stage=f"fusion_{profile['name']}"
            )

            if ret["returncode"] != 0 or not trial_out.exists():
                logger.warning(f"[fusion] Failed: {profile['name']}")
                continue

            try:
                xyz, normals, rgb = _read_ply_full(trial_out)

                valid_mask, scale, spread = _validate_cloud(xyz)

                xyz = xyz[valid_mask]
                normals = normals[valid_mask]
                rgb = rgb[valid_mask]

                filter_mask = _light_filter(xyz)

                xyz = xyz[filter_mask]
                normals = normals[filter_mask]
                rgb = rgb[filter_mask]

                score = _score_cloud(len(xyz), spread)

                logger.info(
                    f"[fusion] {profile['name']} → "
                    f"points={len(xyz)} spread={spread:.2f} score={score:.2f}"
                )

                if score > best_score:
                    best_score = score
                    best_xyz = xyz
                    best_normals = normals
                    best_rgb = rgb
                    best_profile = profile["name"]
                    best_scale = scale
                    best_spread = spread

            except Exception as e:
                logger.warning(
                    f"[fusion] Profile invalid: {profile['name']} → {e}"
                )
                continue

        if best_xyz is None:
            raise RuntimeError("All stereo fusion profiles failed")

        _write_ply_full(
            final_out,
            best_xyz,
            best_normals,
            best_rgb
        )

        _save_metadata(
            meta_out,
            best_profile,
            len(best_xyz),
            best_scale,
            best_spread
        )

        logger.info(f"[fusion] Best profile: {best_profile}")
        logger.info(f"[fusion] Final points: {len(best_xyz)}")

        return {
            "status": "complete",
            "profile": best_profile,
            "points": int(len(best_xyz)),
            "scale": float(best_scale),
            "spread_ratio": float(best_spread),
        }

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)