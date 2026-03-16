from pathlib import Path
from datetime import datetime


class ProjectPaths:
    """
    Central path manager for the photogrammetry pipeline.
    Ensures deterministic directory layout across all runs.
    """

    def __init__(self, project_root: str, run_id: str | None = None):
        self.project_root = Path(project_root).resolve()

        # Input directories
        self.input = self.project_root / "input"
        self.images = self.input / "images"

        # Run directory
        self.runs = self.project_root / "runs"

        if run_id is None:
            run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")

        self.run_id = run_id
        self.run_root = self.runs / self.run_id

        # Stage output directories
        self.sparse = self.run_root / "sparse"
        self.dense = self.run_root / "dense"
        self.mesh = self.run_root / "mesh"
        self.texture = self.run_root / "texture"
        self.evaluation = self.run_root / "evaluation"

        # Utility directories
        self.logs = self.run_root / "logs"
        self.tmp = self.run_root / "tmp"

        # COLMAP specific
        self.colmap = self.sparse / "colmap"
        self.database = self.colmap / "database.db"
        self.sparse_model = self.colmap / "sparse"

        # OpenMVS
        self.openmvs = self.dense / "openmvs"

    def create_all(self):
        """
        Create all directories required for the pipeline run.
        """

        dirs = [
            self.input,
            self.images,
            self.runs,
            self.run_root,
            self.sparse,
            self.dense,
            self.mesh,
            self.texture,
            self.evaluation,
            self.logs,
            self.tmp,
            self.colmap,
            self.sparse_model,
            self.openmvs,
        ]

        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    def summary(self):
        """
        Return dictionary of important paths for debugging/logging.
        """

        return {
            "project_root": self.project_root,
            "run_root": self.run_root,
            "images": self.images,
            "database": self.database,
            "sparse_model": self.sparse_model,
            "dense": self.dense,
            "mesh": self.mesh,
            "texture": self.texture,
            "evaluation": self.evaluation,
        }