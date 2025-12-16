# utils/config.py
import yaml
from pathlib import Path

# Executables
COLMAP_EXE = "colmap"
GLOMAP_EXE = "glomap"
OPENMVS_EXE = "D:\\CSE499_MK-2\\photogram-proj\\OpenMVS_Windows_x64\\OpenMVS.exe"


def load_config(project_root: Path) -> dict:
    """
    Load config.yaml from project root.

    This is the single source of truth for all tunables.
    """
    config_path = project_root / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.yaml at {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f) or {}

    return config
