# utils/config.py
import yaml
import shutil
from pathlib import Path

# -----------------------------
# Executables (PATH-resolved)
# -----------------------------

COLMAP_EXE = "colmap"
GLOMAP_EXE = "glomap"

OPENMVS_TOOLS = {
    "interface_colmap": "InterfaceCOLMAP",
    "densify": "DensifyPointCloud",
    "mesh": "ReconstructMesh",
    "texture": "TextureMesh",
}


def _assert_on_path(exe: str):
    """Fail fast if an executable is not available."""
    if shutil.which(exe) is None:
        raise RuntimeError(f"Executable not found on PATH: {exe}")


def validate_executables():
    """Validate all external tool dependencies."""
    _assert_on_path(COLMAP_EXE)
    _assert_on_path(GLOMAP_EXE)
    for exe in OPENMVS_TOOLS.values():
        _assert_on_path(exe)


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
