from pathlib import Path


class ProjectPaths:
    """
    MARK-2 canonical project layout (OpenMVS-first).

    This class is the ONLY authority for filesystem structure.
    No stage may invent paths outside this contract.
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
        # OpenMVS (authoritative dense backend)
        # -----------------------------
        self.openmvs = self.root / "openmvs"

        # FILE — must NEVER be mkdir'ed
        self.openmvs_scene = self.openmvs / "scene.mvs"

        self.openmvs_dense = self.openmvs / "dense"
        self.openmvs_mesh = self.openmvs / "mesh"
        self.openmvs_texture = self.openmvs / "texture"

        # -----------------------------
        # Tool-agnostic aliases (MANDATORY)
        # -----------------------------
        self.dense = self.openmvs_dense
        self.mesh = self.root / "mesh"
        self.textures = self.root / "textures"
        self.evaluation = self.root / "evaluation"
        self.visualization = self.root / "visualization"
        self.logs = self.root / "logs"
        self.runs = self.root / "runs"

    # ------------------------------------------------------------------
    # Directory creation — STRICTLY directories, NEVER files
    # ------------------------------------------------------------------
    def ensure_all(self):
        directories = [
            self.root,

            # Inputs
            self.raw,
            self.videos,
            self.images_filtered,
            self.images_processed,

            # SfM
            self.database,
            self.sparse,

            # OpenMVS
            self.openmvs,
            self.openmvs_dense,
            self.openmvs_mesh,
            self.openmvs_texture,

            # Tool-agnostic
            self.mesh,
            self.textures,
            self.evaluation,
            self.visualization,
            self.logs,
            self.runs,
        ]

        for d in directories:
            d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Contract validation (structure, not existence)
    # ------------------------------------------------------------------
    def validate(self):
        required_attrs = [
            "dense",
            "openmvs",
            "openmvs_scene",
            "sparse",
            "images_processed",
        ]

        for name in required_attrs:
            if not hasattr(self, name):
                raise RuntimeError(f"ProjectPaths missing attribute: {name}")

        # scene.mvs must be a FILE path, not a directory
        if self.openmvs_scene.exists() and self.openmvs_scene.is_dir():
            raise RuntimeError(
                "openmvs_scene points to a directory, expected a file: scene.mvs"
            )

    def __repr__(self):
        return f"ProjectPaths(root={self.root})"
