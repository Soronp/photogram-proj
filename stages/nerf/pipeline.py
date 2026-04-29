from pathlib import Path
import shutil
import os
import json
import sys


# =====================================================
# ENVIRONMENT
# =====================================================

def _safe_env():
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["RICH_NO_COLOR"] = "1"
    return env


# =====================================================
# EXECUTABLE RESOLUTION
# =====================================================

def _resolve_ns():
    scripts = Path(sys.executable).parent

    train = scripts / "ns-train.exe"
    export = scripts / "ns-export.exe"

    if not train.exists():
        raise RuntimeError(f"ns-train missing → {train}")
    if not export.exists():
        raise RuntimeError(f"ns-export missing → {export}")

    return str(train), str(export)


# =====================================================
# CONFIG DISCOVERY
# =====================================================

def _latest_config(run_root: Path):
    cfgs = list(run_root.rglob("config.yml"))

    if not cfgs:
        raise RuntimeError("No config.yml found in nerf runs")

    return max(cfgs, key=lambda p: p.stat().st_mtime)


# =====================================================
# DATASET VALIDATION
# =====================================================

def _validate_dataset(dataset_dir: Path):
    tf = dataset_dir / "transforms.json"

    if not tf.exists():
        raise RuntimeError("transforms.json missing")

    data = json.loads(tf.read_text())
    frames = data.get("frames", [])

    if len(frames) < 10:
        raise RuntimeError(f"[NERF] Too few frames: {len(frames)}")

    return len(frames)


# =====================================================
# TRAIN COMMAND (FAST VALIDATION MODE)
# =====================================================

def _build_train_cmd(ns_train, dataset_dir, run_root, device):

    return [
        ns_train,
        "nerfacto",

        "--output-dir", str(run_root),
        "--machine.device-type", device,

        # FAST MODE (validation only)
        "--max-num-iterations", "2500",

        "--pipeline.model.predict-normals", "False",
        "--pipeline.model.num-nerf-samples-per-ray", "64",

        "--pipeline.datamanager.train-num-rays-per-batch", "2048",

        "--viewer.quit-on-train-completion", "True",

        "nerfstudio-data",
        "--data", str(dataset_dir),
    ]


def _run_training(ns_train, dataset_dir, run_root, env, tool_runner, logger):

    logger.info("[NERF] Training (FAST, GPU preferred)")

    cmd = _build_train_cmd(ns_train, dataset_dir, run_root, "cuda")

    result = tool_runner.run(
        cmd,
        stage="NERF TRAIN GPU",
        env=env,
        allow_failure=True
    )

    if result["success"]:
        return

    logger.warning("[NERF] GPU failed → CPU fallback")

    cmd = _build_train_cmd(ns_train, dataset_dir, run_root, "cpu")

    tool_runner.run(
        cmd,
        stage="NERF TRAIN CPU",
        env=env,
        allow_failure=False
    )


# =====================================================
# EXPORT POINT CLOUD (AUXILIARY ONLY)
# =====================================================

def _export_pointcloud(ns_export, config, export_dir, env, tool_runner, logger):

    logger.info("[NERF] Exporting auxiliary point cloud")

    cmd = [
        ns_export,
        "pointcloud",

        "--load-config", str(config),
        "--output-dir", str(export_dir),

        "--num-points", "800000",
        "--remove-outliers", "True",
        "--normal-method", "open3d",
    ]

    result = tool_runner.run(
        cmd,
        stage="NERF EXPORT",
        env=env,
        allow_failure=True
    )

    if result["success"]:
        return

    logger.warning("[NERF] Export failed → fallback minimal")

    fallback = [
        ns_export,
        "pointcloud",
        "--load-config", str(config),
        "--output-dir", str(export_dir),
    ]

    tool_runner.run(
        fallback,
        stage="NERF EXPORT FALLBACK",
        env=env,
        allow_failure=False
    )


# =====================================================
# FIND BEST OUTPUT
# =====================================================

def _find_largest_ply(export_dir: Path):
    plys = list(export_dir.rglob("*.ply"))

    if not plys:
        raise RuntimeError("[NERF] No PLY exported")

    return max(plys, key=lambda p: p.stat().st_size)


# =====================================================
# OPTIONAL DEBUG MESH (EXPLICITLY NON-GEOMETRIC)
# =====================================================

def _maybe_debug_mesh(paths, nerf_ply, tool_runner, logger, config):

    if not config.get("nerf", {}).get("debug_mesh", False):
        return None

    debug_mesh = paths.dense / "nerf_debug_mesh.ply"

    logger.info("[NERF] Generating debug mesh (visualization only)")

    cmd = [
        "colmap",
        "poisson_mesher",
        "--input_path", str(nerf_ply),
        "--output_path", str(debug_mesh),
        "--PoissonMeshing.depth", "7",
        "--PoissonMeshing.trim", "7"
    ]

    tool_runner.run(
        cmd,
        stage="NERF DEBUG MESH",
        allow_failure=True
    )

    return debug_mesh


# =====================================================
# MAIN ENTRY
# =====================================================

def run_nerfstudio_dense(paths, config, logger, tool_runner):
    """
    Nerfstudio auxiliary stage.

    Responsibilities:
    - Train NeRF (fast validation mode)
    - Export point cloud (diagnostic / auxiliary)
    - NEVER modify primary geometry outputs
    - NEVER act as mesh authority
    """

    logger.info("==== NERF (AUXILIARY STAGE) ====")

    ns_train, ns_export = _resolve_ns()

    work_dir = paths.dense / "nerf"
    runs_dir = work_dir / "runs"
    export_dir = work_dir / "export"

    runs_dir.mkdir(parents=True, exist_ok=True)
    export_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------
    # DATASET PREP
    # -----------------------
    from stages.nerf.colmap_to_nerf import run_colmap_to_nerfstudio

    dataset_dir = run_colmap_to_nerfstudio(paths, config, logger)
    _validate_dataset(dataset_dir)

    env = _safe_env()

    # -----------------------
    # TRAIN
    # -----------------------
    _run_training(ns_train, dataset_dir, runs_dir, env, tool_runner, logger)

    # -----------------------
    # EXPORT
    # -----------------------
    cfg = _latest_config(runs_dir)
    _export_pointcloud(ns_export, cfg, export_dir, env, tool_runner, logger)

    nerf_ply_src = _find_largest_ply(export_dir)

    # 🔥 STRICT OUTPUT ISOLATION
    nerf_output = paths.dense / "nerf_pointcloud.ply"
    shutil.copy(nerf_ply_src, nerf_output)

    logger.info(f"[NERF] Auxiliary cloud saved → {nerf_output}")

    # -----------------------
    # OPTIONAL DEBUG
    # -----------------------
    debug_mesh = _maybe_debug_mesh(paths, nerf_output, tool_runner, logger, config)

    if debug_mesh:
        logger.info(f"[NERF] Debug mesh → {debug_mesh}")

    # 🚨 CRITICAL: DO NOT RETURN MESH
    return nerf_output