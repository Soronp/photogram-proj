from pathlib import Path


class ProjectPaths:
    """
    Central registry for all MARK-2 project paths.

    No stage should construct paths manually.
    This class is UI-safe and resume-safe.
    """

    def __init__(self, project_root: Path):
        self.root = project_root.resolve()

        # -----------------------------
        # Input & preprocessing
        # -----------------------------
        self.raw = self.root / "raw"                       # ingested data
        self.images = self.root / "images"                 # extracted frames / copied images
        self.images_filtered = self.root / "images_filtered"
        self.images_processed = self.root / "images_processed"

        # -----------------------------
        # Reconstruction
        # -----------------------------
        self.database = self.root / "database"
        self.sparse = self.root / "sparse"
        self.dense = self.root / "dense"

        # -----------------------------
        # Outputs
        # -----------------------------
        self.mesh = self.root / "mesh"
        self.textures = self.root / "textures"
        self.evaluation = self.root / "evaluation"
        self.visualization = self.root / "visualization"

        # -----------------------------
        # Metadata & logs
        # -----------------------------
        self.logs = self.root / "logs"
        self.runs = self.root / "runs"   # run manifests (future-safe)

    def ensure_all(self):
        """Create all required directories."""
        for value in vars(self).values():
            if isinstance(value, Path):
                value.mkdir(parents=True, exist_ok=True)

    def __repr__(self):
        return f"ProjectPaths(root={self.root})"
