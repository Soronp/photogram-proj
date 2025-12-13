import os
import sys
import time
import shutil
import json

from utils.config import PATHS, COLMAP_EXE, GLOMAP_EXE
from utils.logger import get_logger

# Pipeline steps
from input import run_input_handler
from coverage import run_coverage_filter
from pre_proc import run_preprocessing
from reconstruction_core import run_sparse_reconstruction
from dense_reconstruction import run_dense_reconstruction
from gen_mesh import run_mesh_generation
from eval import run_evaluation

logger = get_logger()
CHECKPOINT_FILE = os.path.join(PATHS["logs"], "pipeline_checkpoint.json")


# ---------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------
def check_dependencies():
    """Ensure COLMAP and GLOMAP are present before running pipeline."""
    missing = []
    if not os.path.exists(COLMAP_EXE):
        missing.append(f"COLMAP executable not found: {COLMAP_EXE}")
    if not os.path.exists(GLOMAP_EXE):
        missing.append(f"GLOMAP executable not found: {GLOMAP_EXE} (required as primary sparse solver)")
    if missing:
        for msg in missing:
            logger.error(msg)
        sys.exit(1)
    logger.info(f"Dependencies OK: COLMAP -> {COLMAP_EXE}, GLOMAP -> {GLOMAP_EXE}")


# ---------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------
def load_checkpoint():
    """Load pipeline checkpoint if exists."""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            return json.load(f)
    return {}


def save_checkpoint(step_name, output):
    """Save a step's result to the checkpoint file."""
    os.makedirs(PATHS["logs"], exist_ok=True)
    checkpoint = load_checkpoint()
    checkpoint[step_name] = output
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint, f, indent=2)
    logger.info(f"Checkpoint saved: {step_name}")


# ---------------------------------------------------------------------
# Cleaning pipeline folders
# ---------------------------------------------------------------------
def clean_pipeline_folders():
    """
    Recursively delete contents of all pipeline folders while preserving the folder structure.
    Ensures a clean fresh run.
    """
    folders = [
        PATHS["processed"],
        PATHS["sparse"],
        PATHS["dense"],
        PATHS["mesh"],
        PATHS["evaluations"],
    ]

    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        for item in os.listdir(folder):
            path = os.path.join(folder, item)
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
            except Exception as e:
                logger.warning(f"Failed to remove {path}: {e}")

    if os.path.exists(CHECKPOINT_FILE):
        try:
            os.remove(CHECKPOINT_FILE)
            logger.info("Deleted existing checkpoint (fresh run enforced).")
        except Exception as e:
            logger.warning(f"Could not delete checkpoint file: {e}")

    logger.info("Pipeline folders cleaned successfully.")


# ---------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------
def orchestrate_pipeline():
    """Run the full MARK-2 photogrammetry pipeline with checkpointing."""
    print("=== MARK-2 Photogrammetry Pipeline ===")

    check_dependencies()

    # Decide whether to resume or start fresh
    checkpoint = {}
    if os.path.exists(CHECKPOINT_FILE):
        choice = input("Resume from previous run? [yes/no]: ").strip().lower()
        if choice not in ("yes", "y"):
            logger.info("User selected fresh run. Cleaning folders...")
            clean_pipeline_folders()
        else:
            checkpoint = load_checkpoint()
            logger.info(f"Resuming from checkpoint: {list(checkpoint.keys())}")

    start_total = time.time()
    outputs = {}
    timings = {}

    # Ordered pipeline steps
    pipeline_steps = [
        ("input", run_input_handler),
        ("coverage", run_coverage_filter),
        ("preprocessing", run_preprocessing),
        ("sparse", run_sparse_reconstruction),
        ("dense", run_dense_reconstruction),
        ("mesh", run_mesh_generation),
        ("evaluation", run_evaluation),
    ]

    # Execute steps sequentially with checkpointing
    for step_name, step_func in pipeline_steps:
        if step_name in checkpoint:
            logger.info(f"[SKIP] {step_name} already completed.")
            outputs[step_name] = checkpoint[step_name]
            continue

        logger.info(f"[START] {step_name}")
        t0 = time.time()

        try:
            result = step_func()
            outputs[step_name] = result if result is not None else "done"
            save_checkpoint(step_name, outputs[step_name])
        except Exception as e:
            logger.error(f"[FAIL] {step_name}: {e}")
            logger.error("Pipeline halted due to unrecoverable error.")
            return

        elapsed = time.time() - t0
        timings[step_name] = elapsed
        logger.info(f"[DONE] {step_name} in {elapsed:.2f}s")

    total_time = time.time() - start_total

    # Summary
    logger.info("=== PIPELINE COMPLETE ===")
    logger.info(f"Total runtime: {total_time:.2f}s")
    for step, t in timings.items():
        logger.info(f"{step}: {t:.2f}s")
    logger.info(f"Final outputs: {outputs}")
    logger.info(f"Logs available at: {PATHS['logs']}")


if __name__ == "__main__":
    orchestrate_pipeline()
