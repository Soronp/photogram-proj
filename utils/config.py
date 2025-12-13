import os
from pathlib import Path

# ---------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------
# Core directory layout
# ---------------------------------------------------------------------

PATHS = {
    # Input
    "input_images": PROJECT_ROOT / "input" / "images",
    "input_videos": PROJECT_ROOT / "input" / "videos",

    # Processing stages
    "processed": PROJECT_ROOT / "processed",
    "normalized": PROJECT_ROOT / "processed" / "images_normalized",
    "filtered": PROJECT_ROOT / "processed" / "images_filtered",
    "preprocessed": PROJECT_ROOT / "processed" / "images_preprocessed",

    # Reconstructions
    "recon_root": PROJECT_ROOT / "reconstructions",
    "sparse": PROJECT_ROOT / "reconstructions" / "sparse",
    "dense": PROJECT_ROOT / "reconstructions" / "dense",
    "mesh": PROJECT_ROOT / "reconstructions" / "mesh",

    # Outputs
    "evaluations": PROJECT_ROOT / "evaluations",
    "logs": PROJECT_ROOT / "logs",
}

# ---------------------------------------------------------------------
# External tools
# ---------------------------------------------------------------------
# Use absolute paths if needed, otherwise rely on PATH

COLMAP_EXE = "colmap"
GLOMAP_EXE = "glomap"

# ---------------------------------------------------------------------
# Directory initialization (explicit, predictable)
# ---------------------------------------------------------------------

def ensure_directories():
    """
    Create required project directories.
    Called explicitly by orchestrator — NOT at import time.
    """
    for path in PATHS.values():
        os.makedirs(path, exist_ok=True)
