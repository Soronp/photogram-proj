#!/usr/bin/env python3
"""
High-detail Sparse Reconstruction Pipeline (COLMAP 3.13 SAFE)
- Fully compatible with COLMAP 3.13+
- Maximizes feature coverage where supported
- Uses GLOMAP for global SfM
"""

import os
import subprocess
import json
import shutil
from utils.config import PATHS, COLMAP_EXE, GLOMAP_EXE
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
# Sparse reconstruction
# ------------------------------------------------------------------

def run_sparse_pipeline():
    image_dir = os.path.join(PATHS["processed"], "images_preprocessed")
    sparse_root = PATHS["sparse"]
    db_path = os.path.join(sparse_root, "database.db")
    ply_out = os.path.join(sparse_root, "sparse_model.ply")

    if os.path.exists(sparse_root):
        shutil.rmtree(sparse_root)
    os.makedirs(sparse_root, exist_ok=True)

    # --------------------------------------------------
    # 1. Feature Extraction (MAX QUALITY – VALID FLAGS)
    # --------------------------------------------------
    run_command([
        COLMAP_EXE, "feature_extractor",
        "--database_path", db_path,
        "--image_path", image_dir,
        "--ImageReader.single_camera_per_folder", "0",
        "--SiftExtraction.estimate_affine_shape", "1",
        "--SiftExtraction.domain_size_pooling", "1",
        "--SiftExtraction.max_num_features", "8192"
    ], "Feature Extraction (High Detail)")

    # --------------------------------------------------
    # 2. Exhaustive Matching (NO FLAGS – REQUIRED)
    # --------------------------------------------------
    run_command([
        COLMAP_EXE, "exhaustive_matcher",
        "--database_path", db_path
    ], "Exhaustive Matching (COLMAP 3.13 Safe)")

    # --------------------------------------------------
    # 3. Global Sparse Reconstruction (GLOMAP)
    # --------------------------------------------------
    model_out = os.path.join(sparse_root, "model")
    os.makedirs(model_out, exist_ok=True)

    run_command([
        GLOMAP_EXE, "mapper",
        "--database_path", db_path,
        "--output_path", model_out
    ], "Sparse Reconstruction (GLOMAP)")

    model = find_sparse_model(model_out)
    if model is None:
        raise RuntimeError("Sparse model generation failed")

    # --------------------------------------------------
    # 4. Convert to PLY
    # --------------------------------------------------
    run_command([
        COLMAP_EXE, "model_converter",
        "--input_path", model,
        "--output_path", ply_out,
        "--output_type", "PLY"
    ], "Sparse Model Conversion")

    logger.info(f"Sparse reconstruction complete: {ply_out}")
    return ply_out


# ------------------------------------------------------------------
# Orchestrator
# ------------------------------------------------------------------

def run_sparse_reconstruction():
    cp = load_checkpoint()
    if "sparse" in cp:
        logger.info("Sparse stage already completed. Skipping.")
        return cp["sparse"]

    out = run_sparse_pipeline()
    save_checkpoint("sparse", out)
    return out


if __name__ == "__main__":
    run_sparse_reconstruction()
