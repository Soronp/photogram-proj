from pathlib import Path
from datetime import datetime


class ProjectPaths:
    def __init__(self, project_root: Path, run_name: str = None):
        self.project_root = Path(project_root)

        # -----------------------------
        # RAW (IMMUTABLE INPUT)
        # -----------------------------
        self.raw_images = self.project_root / "raw_images"

        # -----------------------------
        # RUN MANAGEMENT
        # -----------------------------
        self.runs_root = self.project_root / "runs"
        self.runs_root.mkdir(parents=True, exist_ok=True)

        if run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"run_{timestamp}"

        self.run_name = run_name
        self.run_root = self.runs_root / run_name
        self.run_root.mkdir(parents=True, exist_ok=True)

        # -----------------------------
        # WORKING (INGESTION OUTPUT)
        # -----------------------------
        self.working = self.run_root / "working"
        self.images = self.working / "images"
        self.images_downsampled = self.working / "images_downsampled"

        # -----------------------------
        # SPARSE (SfM)
        # -----------------------------
        self.sparse = self.run_root / "sparse"
        self.database = self.sparse / "database.db"

        # 🔥 CRITICAL: canonical model location
        self.sparse_model = self.sparse / "0"

        # -----------------------------
        # DENSE (COLMAP)
        # -----------------------------
        self.dense = self.run_root / "dense"
        self.dense_images = self.dense / "images"
        self.stereo = self.dense / "stereo"
        self.fused = self.dense / "fused.ply"

        # -----------------------------
        # MESH
        # -----------------------------
        self.mesh = self.run_root / "mesh"
        self.mesh_file = self.mesh / "mesh.ply"

        # -----------------------------
        # TEXTURE (OpenMVS)
        # -----------------------------
        self.texture = self.run_root / "texture"

        # -----------------------------
        # LOGS / METRICS
        # -----------------------------
        self.logs = self.run_root / "logs"
        self.log_file = self.logs / "pipeline.log"
        self.metrics = self.run_root / "metrics.json"

        self._create_dirs()

    # --------------------------------------------------
    def _create_dirs(self):
        dirs = [
            self.working,
            self.images,
            self.images_downsampled,
            self.sparse,
            self.dense,
            self.mesh,
            self.texture,
            self.logs,
        ]

        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    def summary(self):
        return {
            "project_root": str(self.project_root),
            "run_root": str(self.run_root),
            "images": str(self.images),
            "images_downsampled": str(self.images_downsampled),
            "sparse_model": str(self.sparse_model),
            "dense": str(self.dense),
        }