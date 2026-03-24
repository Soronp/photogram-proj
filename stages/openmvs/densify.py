from pathlib import Path
import subprocess


def run(paths, config, logger, tool_runner):
    stage = "openmvs_densify"
    logger.info(f"---- {stage.upper()} ----")

    # =====================================================
    # 📁 WORKSPACE
    # =====================================================
    mvs_dir = (paths.run_root / "openmvs").resolve()
    scene = mvs_dir / "scene.mvs"

    if not scene.exists():
        raise RuntimeError(f"{stage}: missing scene.mvs → run export first")

    logger.info(f"{stage}: scene → {scene}")
    logger.info(f"{stage}: workspace → {mvs_dir}")

    # =====================================================
    # ⚙️ CONFIG
    # =====================================================
    cfg = config.get("dense", {}).get("openmvs", {})

    resolution = cfg.get("resolution_level", 1)
    number_views = cfg.get("number_views", 6)
    cuda_device = cfg.get("cuda_device", 0)

    # =====================================================
    # 🧠 COMMAND BUILDER
    # =====================================================
    def build_cmd(device):
        return [
            "DensifyPointCloud",
            "-i", str(scene),
            "-w", str(mvs_dir),

            "--cuda-device", str(device),

            "--resolution-level", str(resolution),
            "--number-views", str(number_views),

            "--fusion-filter", "2",
            "--filter-point-cloud", "1",
            "--estimate-colors", "2",
            "--estimate-normals", "2",
            "--number-views-fuse", "3",
        ]

    # =====================================================
    # 🧪 EXECUTION WITH FULL DEBUG CAPTURE
    # =====================================================
    def run_process(cmd, tag):
        logger.info(f"[{stage}] RUNNING ({tag})")
        logger.info(" ".join(cmd))

        result = subprocess.run(
            cmd,
            cwd=str(mvs_dir),
            capture_output=True,
            text=True
        )

        logger.info(f"[{stage}] EXIT CODE ({tag}) = {result.returncode}")

        if result.stdout:
            logger.info(f"[{stage}] STDOUT ({tag}):\n{result.stdout}")

        if result.stderr:
            logger.error(f"[{stage}] STDERR ({tag}):\n{result.stderr}")

        return result.returncode

    # =====================================================
    # 🚀 GPU FIRST
    # =====================================================
    code = run_process(build_cmd(cuda_device), "GPU")

    # =====================================================
    # 🔥 SMART FAILURE ANALYSIS
    # =====================================================
    if code != 0:
        logger.warning(f"{stage}: GPU FAILED → analyzing issue")

        error_text = openmvs_error_hint = ""

        # simple classification
        if "CUDA" in error_text or "cuda" in error_text:
            openmvs_error_hint = "CUDA/GPU crash or unsupported compute capability"
        elif "scene" in error_text or "input" in error_text:
            openmvs_error_hint = "Invalid or corrupted scene.mvs"
        elif "memory" in error_text or "alloc" in error_text:
            openmvs_error_hint = "Out of memory during densification"
        else:
            openmvs_error_hint = "Unknown OpenMVS internal failure"

        logger.error(f"{stage}: ROOT CAUSE GUESS → {openmvs_error_hint}")

        # =================================================
        # 🧯 CPU FALLBACK (REAL)
        # =================================================
        logger.info(f"{stage}: retrying on CPU (-2)")
        code = run_process(build_cmd(-2), "CPU")

        if code != 0:
            raise RuntimeError(
                f"{stage}: FAILED on BOTH GPU and CPU → check logs above"
            )

    # =====================================================
    # ✅ OUTPUT VALIDATION (REAL CHECK)
    # =====================================================
    dense_scene = mvs_dir / "scene_dense.mvs"

    if not dense_scene.exists() or dense_scene.stat().st_size < 1000:
        raise RuntimeError(
            f"{stage}: densify failed → scene_dense.mvs missing or invalid"
        )

    logger.info(f"{stage}: SUCCESS → {dense_scene}")