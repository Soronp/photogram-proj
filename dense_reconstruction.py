#!/usr/bin/env python3
"""
High-detail Dense Reconstruction (COLMAP 3.13 SAFE)
- Uses valid CLI options only
- Preserves maximum usable resolution
- Stable on Windows
"""

import os
import subprocess
import json
import shutil
from utils.config import PATHS, COLMAP_EXE
from utils.logger import get_logger

logger = get_logger()
CHECKPOINT_FILE = os.path.join(PATHS["logs"], "pipeline_checkpoint.json")


# ------------------------------------------------------------------
# Checkpoint helpers
# ------------------------------------------------------------------

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            return json.load(f)
    return {}

def save_checkpoint(step, output):
    checkpoint = load_checkpoint()
    checkpoint[step] = str(output)
    os.makedirs(PATHS["logs"], exist_ok=True)
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint, f, indent=2)
    logger.info(f"Checkpoint saved: {step}")


# ------------------------------------------------------------------
# Command runner
# ------------------------------------------------------------------

def run_command(cmd, label):
    cmd = [str(c) for c in cmd]
    logger.info(f"[RUN] {label}: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )
    logger.info(result.stdout)
    return result


# ------------------------------------------------------------------
# Sparse model finder
# ------------------------------------------------------------------

def find_sparse_model(base):
    for root, dirs, _ in os.walk(base):
        if "0" in dirs:
            model = os.path.join(root, "0")
            required = ["cameras.bin", "images.bin", "points3D.bin"]
            if all(os.path.exists(os.path.join(model, f)) for f in required):
                return model
    return None


# ------------------------------------------------------------------
# Dense reconstruction
# ------------------------------------------------------------------

def run_dense_colmap():
    image_dir = os.path.join(PATHS["processed"], "images_preprocessed")
    sparse_root = os.path.join(PATHS["sparse"], "model")
    dense_ws = PATHS["dense"]

    sparse_model = find_sparse_model(sparse_root)
    if sparse_model is None:
        raise RuntimeError("Sparse model not found for dense reconstruction")

    if os.path.exists(dense_ws):
        shutil.rmtree(dense_ws)
    os.makedirs(dense_ws, exist_ok=True)

    # --------------------------------------------------
    # 1. Image Undistortion (HIGH DETAIL, SAFE)
    # --------------------------------------------------
    run_command([
        COLMAP_EXE, "image_undistorter",
        "--image_path", image_dir,
        "--input_path", sparse_model,
        "--output_path", dense_ws,
        "--output_type", "COLMAP",
        "--max_image_size", "2800"
    ], "Image Undistortion (High Detail)")

    # --------------------------------------------------
    # 2. PatchMatch Stereo (GEOMETRIC CONSISTENCY)
    # --------------------------------------------------
    run_command([
        COLMAP_EXE, "patch_match_stereo",
        "--workspace_path", dense_ws,
        "--workspace_format", "COLMAP",
        "--PatchMatchStereo.geom_consistency", "1"
    ], "PatchMatch Stereo")

    # --------------------------------------------------
    # 3. Stereo Fusion (MAX DENSITY)
    # --------------------------------------------------
    fused_ply = os.path.join(dense_ws, "fused.ply")
    run_command([
        COLMAP_EXE, "stereo_fusion",
        "--workspace_path", dense_ws,
        "--workspace_format", "COLMAP",
        "--input_type", "geometric",
        "--output_path", fused_ply
    ], "Stereo Fusion")

    if not os.path.exists(fused_ply):
        raise RuntimeError("Dense reconstruction failed: fused.ply missing")

    logger.info(f"Dense reconstruction completed: {fused_ply}")
    return fused_ply


# ------------------------------------------------------------------
# Orchestrator
# ------------------------------------------------------------------

def run_dense_reconstruction():
    cp = load_checkpoint()
    if "dense" in cp:
        logger.info("Dense stage already completed. Skipping.")
        return cp["dense"]

    out = run_dense_colmap()
    save_checkpoint("dense", out)
    return out


if __name__ == "__main__":
    run_dense_reconstruction()
