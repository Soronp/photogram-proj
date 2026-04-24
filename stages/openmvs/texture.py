from pathlib import Path
import subprocess


# =====================================================
# 🔍 VALIDATE INPUTS
# =====================================================

def _validate_inputs(mvs_dir: Path, mesh_input: Path):
    dense_scene = mvs_dir / "scene_dense.mvs"

    if not dense_scene.exists():
        raise RuntimeError("openmvs_texture: missing scene_dense.mvs")

    if not mesh_input.exists():
        raise RuntimeError("openmvs_texture: missing mesh input")

    return dense_scene


# =====================================================
# 🔍 FIND OUTPUT (ROBUST DETECTION)
# =====================================================

def _find_output(texture_dir: Path):
    candidates = [
        texture_dir / "mesh_textured.obj",
        texture_dir / "mesh_textured.ply",
        texture_dir / "mesh.obj",
        texture_dir / "mesh.ply",
    ]

    for c in candidates:
        if c.exists():
            return c

    return None


# =====================================================
# 🚀 MAIN
# =====================================================

def run(paths, config, logger, tool_runner):
    stage = "openmvs_texture"
    logger.info(f"---- {stage.upper()} ----")

    # =====================================================
    # 📁 PATHS
    # =====================================================
    mvs_dir = (paths.run_root / "openmvs").resolve()
    mesh_input = paths.mesh_file.resolve()

    texture_dir = paths.texture
    texture_dir.mkdir(parents=True, exist_ok=True)

    dense_scene = _validate_inputs(mvs_dir, mesh_input)

    # =====================================================
    # ⚙️ CONFIG
    # =====================================================
    cfg = config.get("texture", {})
    pipeline_mode = config.get("pipeline_mode", "default")

    cuda_device = cfg.get("cuda_device", -1)
    resolution_level = cfg.get("resolution_level", 0)

    # =====================================================
    # 🧹 CLEAN OUTPUT
    # =====================================================
    for f in texture_dir.glob("*"):
        f.unlink()

    # =====================================================
    # 🧠 BUILD COMMAND
    # =====================================================
    output_base = texture_dir / "mesh_textured"

    cmd = [
        "TextureMesh",
        "-i", str(dense_scene),
        "-m", str(mesh_input),
        "-o", str(output_base),
        "-w", str(mvs_dir),
        "--cuda-device", str(cuda_device),
    ]

    # =====================================================
    # 🚀 PIPELINE D (HACK MODE)
    # =====================================================
    if pipeline_mode == "D":
        logger.warning(f"[{stage}] PIPELINE D DETECTED → using RELAXED TEXTURING MODE")

        cmd += [
            # 🔥 relax constraints to FORCE projection
            "--min-views", "1",
            "--outlier-threshold", "10",

            # 🔥 less strict resolution matching
            "--resolution-level", "1",

            # 🔥 disable seam enforcement (helps weak geometry)
            "--global-seam-leveling", "0",
            "--local-seam-leveling", "0",

            # 🔥 allow noisier patch selection
            "--cost-smoothness-ratio", "0.1",
            "--patch-packing-heuristic", "3",

            # 🔥 debug-friendly fallback
            "--empty-color", "0",

            "--sharpness-weight", "0.1",
            "--max-texture-size", "8192",
        ]

    # =====================================================
    # ⚙️ DEFAULT (STABLE / HYBRID PIPELINE)
    # =====================================================
    else:
        cmd += [
            "--resolution-level", str(resolution_level),

            "--global-seam-leveling", "0",
            "--local-seam-leveling", "0",

            "--cost-smoothness-ratio", "0.3",
            "--patch-packing-heuristic", "3",

            "--empty-color", "16777215",

            "--sharpness-weight", "0.2",
            "--max-texture-size", "8192",
        ]

    # =====================================================
    # 🚀 EXECUTION
    # =====================================================
    logger.info(f"[{stage}] COMMAND:")
    logger.info(" ".join(cmd))

    result = subprocess.run(
        cmd,
        cwd=str(mvs_dir),
        capture_output=True,
        text=True
    )

    logger.info(f"[{stage}] EXIT CODE = {result.returncode}")

    if result.stdout:
        logger.info(f"[{stage}] STDOUT:\n{result.stdout}")

    if result.stderr:
        logger.error(f"[{stage}] STDERR:\n{result.stderr}")

    if result.returncode != 0:
        raise RuntimeError(f"{stage}: TextureMesh failed")

    # =====================================================
    # ✅ OUTPUT VALIDATION
    # =====================================================
    final_output = _find_output(texture_dir)

    if final_output is None:
        raise RuntimeError(
            f"{stage}: texture failed → no output generated "
            f"(check OpenMVS logs above)"
        )

    # =====================================================
    # 📊 QUALITY CHECK
    # =====================================================
    size_mb = final_output.stat().st_size / (1024 * 1024)

    if size_mb < 1:
        logger.warning(
            f"{stage}: suspiciously small output ({size_mb:.2f} MB) "
            f"→ likely weak projection"
        )

    # 🔥 extra hint for Pipeline D users
    if pipeline_mode == "D":
        logger.warning(
            f"{stage}: Pipeline D uses relaxed texturing → "
            f"result may be noisy or partially incorrect"
        )

    logger.info(f"{stage}: SUCCESS → {final_output}")