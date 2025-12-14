from pathlib import Path


class ProjectPaths:
    """
    Central registry for all project paths.

    No script should construct paths manually.
    """

    def __init__(self, project_root: Path):
        self.root = project_root

        # Core
        self.raw = project_root / "raw"
        self.images = project_root / "images"
        self.images_filtered = project_root / "images_filtered"
        self.images_processed = self.root / "images_processed"

        # Reconstruction
        self.database = project_root / "database"
        self.sparse = project_root / "sparse"
        self.dense = project_root / "dense"

        # Outputs
        self.mesh = project_root / "mesh"
        self.textures = project_root / "textures"
        self.evaluation = project_root / "evaluation"
        self.visualization = project_root / "visualization"


        # Logs
        self.logs = project_root / "logs"

    def ensure_all(self):
        """Create all directories if missing."""
        for path in vars(self).values():
            if isinstance(path, Path):
                path.mkdir(parents=True, exist_ok=True)

    def __repr__(self):
        return f"ProjectPaths(root={self.root})"
