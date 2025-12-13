import os
import json
from utils.config import PATHS
from utils.logger import get_logger

logger = get_logger()

CHECKPOINT_FILE = os.path.join(PATHS['logs'], "pipeline_checkpoint.json")

# Define all pipeline steps in order
PIPELINE_STEPS = [
    "input_handling",
    "coverage_filter",
    "preprocessing",
    "sparse_reconstruction",
    "dense_reconstruction",
    "mesh_generation",
    "evaluation"
]

def initialize_checkpoint():
    """Create an empty checkpoint file if not exists"""
    os.makedirs(PATHS['logs'], exist_ok=True)
    if not os.path.exists(CHECKPOINT_FILE):
        checkpoint_data = {
            "last_completed_step": None,
            "outputs": {}
        }
        save_checkpoint(checkpoint_data)
        logger.info("Initialized pipeline checkpoint.")

def load_checkpoint():
    """Load the checkpoint JSON"""
    if not os.path.exists(CHECKPOINT_FILE):
        initialize_checkpoint()
    with open(CHECKPOINT_FILE, "r") as f:
        data = json.load(f)
    return data

def save_checkpoint(data):
    """Save checkpoint JSON"""
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"Checkpoint saved: last_completed_step = {data.get('last_completed_step')}")

def update_checkpoint(step_name, outputs=None):
    """Update checkpoint after completing a step"""
    if step_name not in PIPELINE_STEPS:
        logger.warning(f"Unknown step: {step_name}")
        return
    checkpoint = load_checkpoint()
    checkpoint['last_completed_step'] = step_name
    if outputs:
        checkpoint['outputs'][step_name] = outputs
    save_checkpoint(checkpoint)

def resume_from_checkpoint():
    """Return the step to resume from"""
    checkpoint = load_checkpoint()
    last_step = checkpoint.get('last_completed_step')
    if last_step is None:
        return PIPELINE_STEPS[0]  # Start from the beginning
    try:
        idx = PIPELINE_STEPS.index(last_step)
        # Resume from next step
        if idx + 1 < len(PIPELINE_STEPS):
            return PIPELINE_STEPS[idx + 1]
        else:
            logger.info("Pipeline already completed.")
            return None
    except ValueError:
        return PIPELINE_STEPS[0]

def get_output(step_name):
    """Get outputs saved for a particular step"""
    checkpoint = load_checkpoint()
    return checkpoint.get('outputs', {}).get(step_name)
