from pathlib import Path
import json
import os


_MARK2_HOME = Path.home() / ".mark2"
_PATHS_CONFIG = _MARK2_HOME / "paths.json"


def _load_runs_root() -> Path | None:
    if _PATHS_CONFIG.exists():
        try:
            with open(_PATHS_CONFIG, "r", encoding="utf-8") as f:
                data = json.load(f)
            return Path(data["runs_root"]).expanduser().resolve()
        except Exception:
            return None
    return None


def _save_runs_root(path: Path):
    _MARK2_HOME.mkdir(parents=True, exist_ok=True)
    with open(_PATHS_CONFIG, "w", encoding="utf-8") as f:
        json.dump({"runs_root": str(path)}, f, indent=2)


def _resolve_runs_root(interactive: bool = False) -> Path:
    existing = _load_runs_root()
    if existing:
        return existing

    if not interactive:
        # Safe fallback if paths.py is imported, not run directly
        default = _MARK2_HOME / "runs"
        _save_runs_root(default)
        return default

    print("\nMARK-2 Run Storage Configuration")
    print("--------------------------------")
    print("No persistent runs directory is configured.")
    print("Please enter a directory where MARK-2 should store all run data.\n")

    while True:
        user_input = input("Runs directory path: ").strip()
        if not user_input:
            print("Path cannot be empty.")
            continue

        path = Path(user_input).expanduser().resolve()
        try:
            path.mkdir(parents=True, exist_ok=True)
            _save_runs_root(path)
            print(f"\nRuns will be stored at:\n  {path}\n")
            return path
        except Exception as exc:
            print(f"Invalid path: {exc}")


class ProjectPaths:
    """
    MARK-2 canonical project layout (OpenMVS-first).

    This class is the SINGLE authority for filesystem structure.
    """

    def __init__(self, project_root: Path):
        self.root = project_root.resolve()

        # -----------------------------
        # Inputs
        # -----------------------------
        self.raw = self.root / "raw"
        self.videos = self.root / "videos"
        self.images_filtered = self.root / "images_filtered"
        self.images_processed = self.root / "images_processed"

        # -----------------------------
        # SfM
        # -----------------------------
        self.database = self.root / "database"
        self.sparse = self.root / "sparse"

        # -----------------------------
        # OpenMVS
        # -----------------------------
        self.openmvs = self.root / "openmvs"
        self.openmvs_scene = self.openmvs / "scene.mvs"
        self.openmvs_dense = self.openmvs / "dense"
        self.openmvs_mesh = self.openmvs / "mesh"
        self.openmvs_texture = self.openmvs / "texture"

        # -----------------------------
        # Tool-agnostic outputs
        # -----------------------------
        self.dense = self.openmvs_dense
        self.mesh = self.root / "mesh"
        self.textures = self.root / "textures"
        self.evaluation = self.root / "evaluation"
        self.visualization = self.root / "visualization"

        # -----------------------------
        # Execution metadata
        # -----------------------------
        self.logs = self.root / "logs"
        self.runs = _resolve_runs_root(interactive=False)

    # ------------------------------------------------------------------
    def ensure_all(self):
        dirs = [
            self.root,
            self.raw,
            self.videos,
            self.images_filtered,
            self.images_processed,
            self.database,
            self.sparse,
            self.openmvs,
            self.openmvs_dense,
            self.openmvs_mesh,
            self.openmvs_texture,
            self.mesh,
            self.textures,
            self.evaluation,
            self.visualization,
            self.logs,
            self.runs,
        ]

        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def validate(self):
        if self.openmvs_scene.exists() and self.openmvs_scene.is_dir():
            raise RuntimeError(
                "openmvs_scene refers to a directory; expected file path: scene.mvs"
            )

    def __repr__(self):
        return f"ProjectPaths(root={self.root}, runs={self.runs})"


# ----------------------------------------------------------------------
# Interactive configuration entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    _resolve_runs_root(interactive=True)
