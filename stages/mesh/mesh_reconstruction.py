from pathlib import Path
import shutil


# =====================================================
# EXEC WRAPPER
# =====================================================
def _run(tool_runner, cmd, logger, stage, allow_fail=False):
    logger.info(f"[{stage}] CMD: {' '.join(cmd)}")
    return tool_runner.run(cmd, stage=stage, allow_failure=allow_fail)


# =====================================================
# OPTIONAL OPEN3D
# =====================================================
def _get_o3d(logger):
    try:
        import open3d as o3d
        return o3d
    except ImportError:
        logger.warning("[mesh] Open3D not available")
        return None


# =====================================================
# COLMAP MESHING
# =====================================================
def _mesh_colmap(paths, tool_runner, logger):
    fused = paths.dense / "fused.ply"

    if not fused.exists():
        raise RuntimeError("[mesh] COLMAP requires fused.ply")

    out = paths.dense / "mesh_poisson.ply"

    cmd = [
        "colmap", "poisson_mesher",
        "--input_path", str(fused),
        "--output_path", str(out)
    ]

    ret = _run(tool_runner, cmd, logger, "colmap_poisson", allow_fail=True)

    if ret["returncode"] != 0 or not out.exists():
        raise RuntimeError("[mesh] COLMAP meshing failed")

    return out


# =====================================================
# OPENMVS MESHING
# =====================================================
def _mesh_openmvs(paths, tool_runner, logger):
    mvs = paths.run_root / "openmvs" / "scene_dense.mvs"
    out = paths.run_root / "openmvs" / "mesh.ply"

    if not mvs.exists():
        raise RuntimeError("[mesh] Missing scene_dense.mvs")

    cmd = [
        "ReconstructMesh",
        "-i", str(mvs),
        "-o", str(out)
    ]

    ret = _run(tool_runner, cmd, logger, "openmvs_mesh")

    if ret["returncode"] != 0 or not out.exists():
        raise RuntimeError("[mesh] OpenMVS meshing failed")

    return out


# =====================================================
# NERFSTUDIO HANDLING
# =====================================================
def _mesh_nerfstudio(paths, config, tool_runner, logger):
    ns_cfg = config["dense"]["nerfstudio"]
    export_type = ns_cfg.get("export", {}).get("type", "pointcloud")

    fused = paths.dense / "fused.ply"

    # Case 1: direct mesh export
    if export_type == "mesh":
        if not fused.exists():
            raise RuntimeError("[mesh] Nerfstudio mesh export missing")
        logger.info("[mesh] Using Nerfstudio mesh directly")
        return fused

    # Case 2: pointcloud → fallback to COLMAP meshing
    logger.info("[mesh] Nerfstudio pointcloud → COLMAP meshing fallback")
    return _mesh_colmap(paths, tool_runner, logger)


# =====================================================
# CLEANUP (OPTIONAL)
# =====================================================
def _cleanup(mesh_path, out_path, logger):
    o3d = _get_o3d(logger)
    if o3d is None:
        return mesh_path

    logger.info("[mesh] Cleaning mesh")

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))

    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()

    mesh = mesh.filter_smooth_taubin(number_of_iterations=1)

    o3d.io.write_triangle_mesh(str(out_path), mesh)

    return out_path


# =====================================================
# QUALITY SCORE (SIMPLE + SAFE)
# =====================================================
def _score(mesh_path, logger):
    o3d = _get_o3d(logger)
    if o3d is None:
        return 0

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))

    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        return 0

    v = len(mesh.vertices)
    t = len(mesh.triangles)

    density = t / max(v, 1)
    return v * (1.0 / (1.0 + abs(density - 2.0)))


# =====================================================
# MAIN ENTRY
# =====================================================
def run(paths, config, logger, tool_runner):
    logger.info("==== MESH STAGE (ROBUST REWRITE) ====")

    backend = config["pipeline"]["backends"].get("dense")
    final_mesh = paths.mesh_file

    # -------------------------------------------------
    # BACKEND DISPATCH
    # -------------------------------------------------
    if backend == "colmap":
        mesh = _mesh_colmap(paths, tool_runner, logger)

    elif backend == "openmvs":
        mesh = _mesh_openmvs(paths, tool_runner, logger)

    elif backend == "nerfstudio":
        mesh = _mesh_nerfstudio(paths, config, tool_runner, logger)

    else:
        raise RuntimeError(f"[mesh] Unsupported backend: {backend}")

    # -------------------------------------------------
    # CLEANUP
    # -------------------------------------------------
    clean_path = paths.run_root / "mesh_clean.ply"
    mesh = _cleanup(mesh, clean_path, logger)

    # -------------------------------------------------
    # FINALIZE
    # -------------------------------------------------
    shutil.copy(str(mesh), str(final_mesh))

    logger.info(f"[mesh] FINAL → {final_mesh}")

    return {
        "status": "complete",
        "mesh": str(final_mesh),
        "backend": backend,
        "score": _score(mesh, logger)
    }