from pathlib import Path
import open3d as o3d
import numpy as np


def _run_fusion(tool_runner, dense_dir, out, p):
    tool_runner.run([
        "colmap", "stereo_fusion",
        "--workspace_path", str(dense_dir),
        "--workspace_format", "COLMAP",
        "--input_type", "geometric",
        "--output_path", str(out),

        "--StereoFusion.min_num_pixels", str(p["min_pixels"]),
        "--StereoFusion.max_reproj_error", str(p["max_reproj"]),
        "--StereoFusion.max_depth_error", str(p["max_depth"]),
        "--StereoFusion.max_normal_error", str(p["max_normal"]),
        "--StereoFusion.max_traversal_depth", str(p["traversal"]),
        "--StereoFusion.num_threads", "1",
    ])


def _load(path):
    pcd = o3d.io.read_point_cloud(str(path))
    return pcd if len(pcd.points) > 0 else None


def _merge_all(pcds):
    pts = []
    for p in pcds:
        if p:
            pts.append(np.asarray(p.points))
    merged = np.vstack(pts)

    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(merged)
    return out


def _post(pcd):
    # 🔥 LESS DESTRUCTIVE
    pcd, _ = pcd.remove_statistical_outlier(20, 2.5)
    pcd = pcd.voxel_down_sample(0.002)
    pcd.estimate_normals()
    return pcd


def run(paths, config, logger, tool_runner):
    logger.info("==== STEREO FUSION (HOLE-FREE MODE) ====")

    dense = paths.dense

    # =================================================
    # PASS 1 — CLEAN CORE
    # =================================================
    core = dense / "fused_core.ply"
    _run_fusion(tool_runner, dense, core, {
        "min_pixels": 3,
        "max_reproj": 1.5,
        "max_depth": 0.01,
        "max_normal": 12,
        "traversal": 80,
    })

    # =================================================
    # PASS 2 — HOLE FILL
    # =================================================
    fill = dense / "fused_fill.ply"
    _run_fusion(tool_runner, dense, fill, {
        "min_pixels": 2,
        "max_reproj": 3.0,
        "max_depth": 0.03,
        "max_normal": 25,
        "traversal": 150,
    })

    # =================================================
    # PASS 3 — EXTENSION (CRITICAL FIX)
    # =================================================
    extend = dense / "fused_extend.ply"
    _run_fusion(tool_runner, dense, extend, {
        "min_pixels": 1,
        "max_reproj": 4.0,
        "max_depth": 0.05,
        "max_normal": 35,
        "traversal": 200,
    })

    # =================================================
    # MERGE (NO OVER-REJECTION)
    # =================================================
    pcd = _merge_all([
        _load(core),
        _load(fill),
        _load(extend)
    ])

    pcd = _post(pcd)

    final = dense / "fused.ply"
    o3d.io.write_point_cloud(str(final), pcd)

    logger.info(f"Final points: {len(pcd.points)}")

    return {"status": "complete_fusion"}