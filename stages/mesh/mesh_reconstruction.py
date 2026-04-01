from pathlib import Path
import json


# =====================================================
# METADATA
# =====================================================
def _load_metadata(path):
    if not path.exists():
        raise RuntimeError("Missing fusion_metadata.json")

    with open(path, "r") as f:
        return json.load(f)


# =====================================================
# SAFE TOOL EXECUTION (MATCHES YOUR TOOLRUNNER)
# =====================================================
def _run(tool_runner, cmd, logger, stage, allow_fail=False):
    ret = tool_runner.run(
        cmd,
        stage=stage,
        allow_failure=allow_fail
    )

    return ret


# =====================================================
# DELAUNAY MESHING
# =====================================================
def _run_delaunay(paths, tool_runner, logger):
    out_path = paths.dense / "mesh_delaunay.ply"

    cmd = [
        "colmap", "delaunay_mesher",
        "--input_path", str(paths.dense),
        "--output_path", str(out_path)
    ]

    logger.info("[mesh] Running DELAUNAY")

    ret = _run(tool_runner, cmd, logger, "delaunay", allow_fail=True)

    if ret["returncode"] != 0 or not out_path.exists():
        logger.warning("[mesh] Delaunay failed")
        return None

    return out_path


# =====================================================
# POISSON MESHING
# =====================================================
def _run_poisson(paths, tool_runner, logger):
    out_path = paths.dense / "mesh_poisson.ply"

    cmd = [
        "colmap", "poisson_mesher",
        "--input_path", str(paths.dense / "fused.ply"),
        "--output_path", str(out_path),
        "--PoissonMeshing.trim", "10",
        "--PoissonMeshing.depth", "11"
    ]

    logger.info("[mesh] Running POISSON")

    ret = _run(tool_runner, cmd, logger, "poisson", allow_fail=True)

    if ret["returncode"] != 0 or not out_path.exists():
        logger.warning("[mesh] Poisson failed")
        return None

    return out_path


# =====================================================
# VALIDATION (FILE-LEVEL, FAST)
# =====================================================
def _validate_mesh(path, logger):
    try:
        size = path.stat().st_size

        if size < 10000:
            raise RuntimeError("Too small")

        logger.info(f"[mesh] Valid mesh ({size/1e6:.2f} MB)")
        return True

    except Exception as e:
        logger.warning(f"[mesh] Invalid mesh: {e}")
        return False


# =====================================================
# SELECT BEST RESULT
# =====================================================
def _select_mesh(delaunay, poisson, final_path, logger):
    # Prefer Poisson
    if poisson and _validate_mesh(poisson, logger):
        logger.info("[mesh] Using POISSON result")
        poisson.replace(final_path)
        return

    # Fallback to Delaunay
    if delaunay and _validate_mesh(delaunay, logger):
        logger.info("[mesh] Using DELAUNAY fallback")
        delaunay.replace(final_path)
        return

    raise RuntimeError("No valid mesh produced")


# =====================================================
# MAIN ENTRY (FIXED SIGNATURE)
# =====================================================
def run(paths, config, logger, tool_runner):
    logger.info("==== MESH STAGE (COLMAP HYBRID STABLE) ====")

    fused = paths.dense / "fused.ply"
    meta_path = paths.dense / "fusion_metadata.json"
    final_mesh = paths.mesh_file

    if not fused.exists():
        raise RuntimeError("Missing fused.ply")

    _load_metadata(meta_path)

    # -------------------------------------------------
    # RUN BOTH METHODS (SAFE)
    # -------------------------------------------------
    delaunay_mesh = _run_delaunay(paths, tool_runner, logger)
    poisson_mesh = _run_poisson(paths, tool_runner, logger)

    # -------------------------------------------------
    # SELECT BEST
    # -------------------------------------------------
    _select_mesh(delaunay_mesh, poisson_mesh, final_mesh, logger)

    logger.info(f"[mesh] FINAL OUTPUT → {final_mesh}")

    return {
        "status": "complete",
        "mesh": str(final_mesh)
    }