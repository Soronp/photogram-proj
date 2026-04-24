from pathlib import Path
import shutil
import os
import json
import sys


# =====================================================
# ENV
# =====================================================

def _safe_env():
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["RICH_NO_COLOR"] = "1"
    return env


# =====================================================
# EXECUTABLES
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
        raise RuntimeError("No config.yml found")

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
        raise RuntimeError(f"Too few frames: {len(frames)}")

    return len(frames)


# =====================================================
# TRAIN (FAST MODE)
# =====================================================

def _build_cmd(ns_train, dataset_dir, run_root, device):

    return [
        ns_train,
        "nerfacto",

        "--output-dir", str(run_root),
        "--machine.device-type", device,

        # 🔥 FAST TEST MODE
        "--max-num-iterations", "3000",

        "--pipeline.model.predict-normals", "False",
        "--pipeline.model.num-nerf-samples-per-ray", "64",

        "--pipeline.datamanager.train-num-rays-per-batch", "2048",

        "--viewer.quit-on-train-completion", "True",

        "nerfstudio-data",
        "--data", str(dataset_dir),
    ]


def _run_training(ns_train, dataset_dir, run_root, env, tool_runner, logger):

    logger.info("[NERF] FAST training (GPU)")

    cmd = _build_cmd(ns_train, dataset_dir, run_root, "cuda")

    result = tool_runner.run(
        cmd,
        stage="NERF TRAIN GPU",
        env=env,
        allow_failure=True
    )

    if result["success"]:
        return

    logger.warning("[NERF] GPU failed → CPU fallback")

    cmd = _build_cmd(ns_train, dataset_dir, run_root, "cpu")

    tool_runner.run(
        cmd,
        stage="NERF TRAIN CPU",
        env=env,
        allow_failure=False
    )


# =====================================================
# EXPORT (STABLE)
# =====================================================

def _export(ns_export, config, export_dir, env, tool_runner, logger):

    logger.info("[NERF] Exporting point cloud (stable mode)")

    cmd = [
        ns_export,
        "pointcloud",

        "--load-config", str(config),
        "--output-dir", str(export_dir),

        "--num-points", "1000000",   # 🔥 lower for speed
        "--remove-outliers", "True",
        "--normal-method", "open3d",  # 🔥 stable (NOT model_output)
    ]

    result = tool_runner.run(
        cmd,
        stage="NERF EXPORT",
        env=env,
        allow_failure=True
    )

    if result["success"]:
        return

    # fallback (even simpler)
    logger.warning("[NERF] Export failed → retry minimal")

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
# FIND PLY
# =====================================================

def _find_ply(export_dir: Path):
    plys = list(export_dir.rglob("*.ply"))

    if not plys:
        raise RuntimeError("No PLY exported")

    return max(plys, key=lambda p: p.stat().st_size)


# =====================================================
# MESH (FAST POISSON)
# =====================================================

def _mesh(paths, tool_runner, logger):

    logger.info("[NERF] Poisson meshing (fast mode)")

    input_ply = paths.dense / "fused.ply"
    output_mesh = paths.dense / "final_mesh.ply"

    cmd = [
        "colmap",
        "poisson_mesher",
        "--input_path", str(input_ply),
        "--output_path", str(output_mesh),

        # 🔥 FAST SETTINGS
        "--PoissonMeshing.depth", "8",
        "--PoissonMeshing.trim", "6"
    ]

    tool_runner.run(
        cmd,
        stage="POISSON MESH",
        allow_failure=False
    )

    return output_mesh


# =====================================================
# MAIN
# =====================================================

def run_nerfstudio_dense(paths, config, logger, tool_runner):

    logger.info("==== FAST NERF → MESH PIPELINE ====")

    ns_train, ns_export = _resolve_ns()

    work = paths.dense / "nerf"
    runs = work / "runs"
    export_dir = work / "export"

    runs.mkdir(parents=True, exist_ok=True)
    export_dir.mkdir(parents=True, exist_ok=True)

    from stages.nerf.colmap_to_nerf import run_colmap_to_nerfstudio

    dataset = run_colmap_to_nerfstudio(paths, config, logger)
    _validate_dataset(dataset)

    env = _safe_env()

    # -----------------------
    # TRAIN
    # -----------------------
    _run_training(ns_train, dataset, runs, env, tool_runner, logger)

    # -----------------------
    # EXPORT
    # -----------------------
    cfg = _latest_config(runs)
    _export(ns_export, cfg, export_dir, env, tool_runner, logger)

    # -----------------------
    # FINALIZE
    # -----------------------
    ply = _find_ply(export_dir)

    fused = paths.dense / "fused.ply"
    shutil.copy(ply, fused)

    # -----------------------
    # MESH
    # -----------------------
    final_mesh = _mesh(paths, tool_runner, logger)

    logger.info(f"[NERF] FINAL MESH → {final_mesh}")

    return final_mesh