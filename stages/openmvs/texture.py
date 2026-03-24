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

    cuda_device = cfg.get("cuda_device", -1)
    resolution_level = cfg.get("resolution_level", 0)

    # =====================================================
    # 🧹 CLEAN PREVIOUS OUTPUTS (IMPORTANT)
    # =====================================================
    for f in texture_dir.glob("*"):
        f.unlink()

    # =====================================================
    # 🚀 COMMAND (WORKSPACE-CORRECT)
    # =====================================================
    # 🔥 IMPORTANT:
    # Let OpenMVS decide format based on mesh input
    # Do NOT force only OBJ

    output_base = texture_dir / "mesh_textured"

    cmd = [
        "TextureMesh",

        "-i", str(dense_scene),
        "-m", str(mesh_input),

        # base name only (no extension assumption)
        "-o", str(output_base),

        "-w", str(mvs_dir),

        "--cuda-device", str(cuda_device),

        # =====================================================
        # 🎨 STABLE PARAMETERS (POST-FIX BASELINE)
        # =====================================================
        "--resolution-level", str(resolution_level),

        "--global-seam-leveling", "0",
        "--local-seam-leveling", "0",

        "--cost-smoothness-ratio", "0.3",
        "--patch-packing-heuristic", "3",

        "--empty-color", "16777215",  # white fallback

        "--sharpness-weight", "0.2",
        "--max-texture-size", "8192",
    ]

    logger.info(f"[{stage}] COMMAND:")
    logger.info(" ".join(cmd))

    # =====================================================
    # ▶️ EXECUTION
    # =====================================================
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
    # ✅ OUTPUT RESOLUTION (CRITICAL FIX)
    # =====================================================
    final_output = _find_output(texture_dir)

    if final_output is None:
        raise RuntimeError(
            f"{stage}: texture failed → no output generated "
            f"(check OpenMVS logs above)"
        )

    # =====================================================
    # 📊 EXTRA VALIDATION (OPTIONAL BUT STRONG)
    # =====================================================
    size_mb = final_output.stat().st_size / (1024 * 1024)

    if size_mb < 1:
        logger.warning(f"{stage}: suspiciously small output ({size_mb:.2f} MB)")

    logger.info(f"{stage}: SUCCESS → {final_output}")