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
    # ⚙️ BASE CONFIG
    # =====================================================
    cfg = config.get("dense", {}).get("openmvs", {})

    resolution = cfg.get("resolution_level", 1)
    number_views = cfg.get("number_views", 6)
    cuda_device = cfg.get("cuda_device", 0)

    pipeline_mode = config.get("pipeline_mode", "default")

    # =====================================================
    # 🧠 PARAMETER STRATEGIES (ADAPTIVE)
    # =====================================================
    STRATEGIES = [
        # Attempt 1: strict (default / high quality)
        {
            "name": "strict",
            "args": {
                "--resolution-level": str(resolution),
                "--number-views": str(number_views),
                "--number-views-fuse": "3",
                "--fusion-filter": "2",
                "--filter-point-cloud": "1",
                "--estimate-colors": "2",
                "--estimate-normals": "2",
            },
        },

        # Attempt 2: relaxed views + less filtering
        {
            "name": "relaxed_views",
            "args": {
                "--resolution-level": str(resolution),
                "--number-views": "4",
                "--number-views-fuse": "3",
                "--fusion-filter": "1",
                "--filter-point-cloud": "0",
                "--estimate-colors": "2",
                "--estimate-normals": "2",
            },
        },

        # Attempt 3: very robust / fallback
        {
            "name": "robust",
            "args": {
                "--resolution-level": "0",
                "--number-views": "3",
                "--number-views-fuse": "2",
                "--fusion-filter": "0",
                "--filter-point-cloud": "0",
                "--estimate-colors": "2",
                "--estimate-normals": "2",
                "--min-resolution": "640",
            },
        },
    ]

    # =====================================================
    # 🧠 COMMAND BUILDER
    # =====================================================
    def build_cmd(device, strategy_args):
        cmd = [
            "DensifyPointCloud",
            "-i", str(scene),
            "-w", str(mvs_dir),
            "--cuda-device", str(device),
        ]

        # Add strategy-specific args
        for k, v in strategy_args.items():
            cmd += [k, v]

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
    # 🚀 ADAPTIVE EXECUTION LOOP
    # =====================================================
    success = False
    dense_scene = mvs_dir / "scene_dense.mvs"

    for strategy in STRATEGIES:
        logger.info(f"[{stage}] محاولة strategy = {strategy['name']}")

        # Try GPU first
        cmd_gpu = build_cmd(cuda_device, strategy["args"])
        code = run_process(cmd_gpu, f"{strategy['name']}_GPU")

        # GPU fallback to CPU if needed
        if code != 0:
            logger.warning(f"{stage}: GPU failed → retry CPU")

            cmd_cpu = build_cmd(-2, strategy["args"])
            code = run_process(cmd_cpu, f"{strategy['name']}_CPU")

        # Check output
        if dense_scene.exists() and dense_scene.stat().st_size > 1000 and code == 0:
            logger.info(f"{stage}: SUCCESS with strategy '{strategy['name']}'")
            success = True
            break
        else:
            logger.warning(f"{stage}: strategy '{strategy['name']}' failed or weak output")

    # =====================================================
    # ❌ FINAL FAILURE
    # =====================================================
    if not success:
        raise RuntimeError(f"{stage}: FAILED after all adaptive strategies")

    logger.info(f"{stage}: OUTPUT → {dense_scene}")