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

    # 🔥 PIPELINE MODE SWITCH
    pipeline_mode = config.get("pipeline_mode", "default")

    # =====================================================
    # 🧠 COMMAND BUILDER
    # =====================================================
    def build_cmd(device):
        cmd = [
            "DensifyPointCloud",
            "-i", str(scene),
            "-w", str(mvs_dir),

            "--cuda-device", str(device),
        ]

        # =================================================
        # 🚀 PIPELINE D (MAX DETAIL / OPENMVG-FRIENDLY)
        # =================================================
        if pipeline_mode == "D":
            logger.info(f"[{stage}] Using Pipeline D (high detail mode)")

            cmd += [
                "--resolution-level", "0",        # 🔥 full resolution
                "--number-views", "8",            # 🔥 more neighbors
                "--number-views-fuse", "5",

                "--fusion-filter", "0",           # 🔥 NO aggressive filtering
                "--filter-point-cloud", "0",      # 🔥 keep weak points

                "--estimate-colors", "2",
                "--estimate-normals", "2",

                "--min-resolution", "640",        # 🔥 prevent over-downscale
                "--max-resolution", "6000",

                "--sub-resolution-levels", "2",   # 🔥 multi-scale depth
            ]

        # =================================================
        # ⚙️ DEFAULT PIPELINE (UNCHANGED)
        # =================================================
        else:
            cmd += [
                "--resolution-level", str(resolution),
                "--number-views", str(number_views),

                "--fusion-filter", "2",
                "--filter-point-cloud", "1",
                "--estimate-colors", "2",
                "--estimate-normals", "2",
                "--number-views-fuse", "3",
            ]

        return cmd

    # =====================================================
    # 🧪 EXECUTION
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
    # 🔁 CPU FALLBACK
    # =====================================================
    if code != 0:
        logger.warning(f"{stage}: GPU FAILED → retrying on CPU")
        code = run_process(build_cmd(-2), "CPU")

        if code != 0:
            raise RuntimeError(
                f"{stage}: FAILED on BOTH GPU and CPU → check logs above"
            )

    # =====================================================
    # ✅ OUTPUT VALIDATION
    # =====================================================
    dense_scene = mvs_dir / "scene_dense.mvs"

    if not dense_scene.exists() or dense_scene.stat().st_size < 1000:
        raise RuntimeError(
            f"{stage}: densify failed → scene_dense.mvs missing or invalid"
        )

    logger.info(f"{stage}: SUCCESS → {dense_scene}")