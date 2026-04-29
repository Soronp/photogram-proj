from pathlib import Path


class ProjectPaths:
    """
    🔥 FIXED DESIGN:
    - NO automatic run creation
    - NO timestamped folders
    - run_root is ALWAYS externally controlled
    - supports resume-safe deterministic execution
    """

    def __init__(self, run_root: Path):
        self.run_root = Path(run_root).resolve()

        if not self.run_root.exists():
            raise FileNotFoundError(
                f"Run root does not exist: {self.run_root}"
            )

        # =====================================================
        # INPUT DATA (OPTIONAL COPY OR EXTERNAL)
        # =====================================================
        self.raw_images = self.run_root / "raw_images"

        # =====================================================
        # CORE WORK DIRECTORIES
        # =====================================================
        self.working = self.run_root / "working"

        self.images = self.working / "images"
        self.images_downsampled = self.working / "images_downsampled"

        # =====================================================
        # SPARSE RECONSTRUCTION (SfM)
        # =====================================================
        self.sparse = self.run_root / "sparse"
        self.database = self.sparse / "database.db"
        self.sparse_model = self.sparse / "0"  # COLMAP standard output

        # =====================================================
        # DENSE RECONSTRUCTION
        # =====================================================
        self.dense = self.run_root / "dense"
        self.dense_images = self.dense / "images"
        self.stereo = self.dense / "stereo"
        self.fused = self.dense / "fused.ply"

        # =====================================================
        # OPENMVS (if used)
        # =====================================================
        self.openmvs = self.run_root / "openmvs"
        self.openmvs_scene = self.openmvs / "scene_dense.mvs"

        # =====================================================
        # MESH OUTPUT
        # =====================================================
        self.mesh = self.run_root / "mesh"
        self.mesh_file = self.mesh / "mesh.ply"

        # =====================================================
        # TEXTURE OUTPUT
        # =====================================================
        self.texture = self.run_root / "texture"

        # =====================================================
        # LOGGING / METRICS / STATE
        # =====================================================
        self.logs = self.run_root / "logs"
        self.log_file = self.logs / "pipeline.log"

        self.metrics_file = self.run_root / "metrics.json"
        self.state_file = self.run_root / "pipeline_state.json"

        # =====================================================
        # CREATE DIRECTORIES (SAFE, NO SIDE EFFECTS OUTSIDE RUN)
        # =====================================================
        self._create_dirs()

    # =====================================================
    def _create_dirs(self):
        dirs = [
            self.working,
            self.images,
            self.images_downsampled,
            self.sparse,
            self.dense,
            self.openmvs,
            self.mesh,
            self.texture,
            self.logs,
        ]

        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    # =====================================================
    def summary(self):
        return {
            "run_root": str(self.run_root),
            "raw_images": str(self.raw_images),
            "working": str(self.working),
            "sparse": str(self.sparse),
            "dense": str(self.dense),
            "openmvs": str(self.openmvs),
            "mesh": str(self.mesh),
            "texture": str(self.texture),
        }