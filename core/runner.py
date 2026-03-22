from copy import deepcopy

from utils.paths import ProjectPaths
from utils.logger import setup_logger
from core.tool_runner import ToolRunner

# INGESTION
from stages.ingestion.ingest_images import run as ingest_images
from stages.ingestion.validate_images import run as validate_images
from stages.ingestion.downsample import run as downsample_images

# SPARSE
from stages.sparse.feature_extraction import run as feature_extraction
from stages.sparse.feature_matching import run as matching
from stages.sparse.mapper import run as mapper

# COLMAP DENSE
from stages.dense.colmap.image_undistorter import run as undistort
from stages.dense.colmap.patch_match_stereo import run as patch_match
from stages.dense.colmap.stereo_fusion import run as stereo_fusion

# OPENMVS DENSE
from stages.openmvs.export_openmvs import run as openmvs_export
from stages.openmvs.densify import run as openmvs_densify
from stages.openmvs.mesh import run as openmvs_mesh
from stages.openmvs.texture import run as openmvs_texture

# MESH (COLMAP/Open3D)
from stages.mesh.mesh_reconstruction import run as mesh_reconstruction


class PipelineRunner:

    def __init__(self, config, pipeline_type="A"):
        self.pipeline_type = pipeline_type

        project_root = config["paths"]["project_root"]
        self.paths = ProjectPaths(project_root)
        self.logger = setup_logger(self.paths.log_file)

        self.base_config = deepcopy(config)
        self.config = deepcopy(config)
        self.tool_runner = ToolRunner(self.logger)

        self.max_adjustments = 2

        self._configure_pipeline()

    # =====================================================
    # PIPELINE CONFIGURATION
    # =====================================================
    def _configure_pipeline(self):
        if self.pipeline_type == "A":
            self.logger.info("Pipeline A: COLMAP full")
            self.config["sparse"]["backend"] = "colmap"
            self.config["pipeline"]["dense_backend"] = "colmap"

        elif self.pipeline_type == "B":
            self.logger.info("Pipeline B: GLOMAP + COLMAP dense")
            self.config["sparse"]["backend"] = "glomap"
            self.config["pipeline"]["dense_backend"] = "colmap"

        elif self.pipeline_type == "C":
            self.logger.info("Pipeline C: COLMAP + OpenMVS")
            self.config["sparse"]["backend"] = "colmap"
            self.config["pipeline"]["dense_backend"] = "openmvs"
            self.config["pipeline"]["mesh_backend"] = "openmvs"
            self.config["pipeline"]["texture_backend"] = "openmvs"

        else:
            raise ValueError(f"Invalid pipeline type: {self.pipeline_type}")

    # =====================================================
    # MAIN RUN
    # =====================================================
    def run(self):
        self.logger.info("========== PIPELINE START ==========")

        self._run_ingestion()
        self._run_feature_pipeline()
        self._force_high_quality_start()

        # -------------------------
        # SPARSE LOOP
        # -------------------------
        for i in range(self.max_adjustments + 1):
            self.logger.info(f"===== SPARSE PASS {i+1} =====")
            self._run_mapper()

            status = self._evaluate_sparse()
            if status == "good":
                self.logger.info("✅ Sparse reconstruction stable")
                break
            elif status == "too_low":
                self._increase_quality()
            elif status == "too_high":
                self._decrease_quality()

        # -------------------------
        # DENSE
        # -------------------------
        self.logger.info("===== DENSE RECONSTRUCTION =====")
        self._run_dense()

        # -------------------------
        # MESH
        # -------------------------
        self._run_mesh()

        # -------------------------
        # TEXTURE
        # -------------------------
        self._run_texture()

        self.logger.info("========== PIPELINE END ==========")

    # =====================================================
    # INGESTION
    # =====================================================
    def _run_ingestion(self):
        ingest_images(self.paths, self.config, self.logger)
        validate_images(self.paths, self.config, self.logger)
        if self.config.get("downsampling", {}).get("enabled", False):
            downsample_images(self.paths, self.config, self.logger)

    # =====================================================
    # FEATURES
    # =====================================================
    def _run_feature_pipeline(self):
        feature_extraction(self.paths, self.config, self.logger, self.tool_runner)
        matching(self.paths, self.config, self.logger, self.tool_runner)

    # =====================================================
    # SPARSE
    # =====================================================
    def _run_mapper(self):
        mapper(self.paths, self.config, self.logger, self.tool_runner)

    def _evaluate_sparse(self):
        pts_file = self.paths.sparse_model / "points3D.bin"

        if not pts_file.exists():
            return "too_low"

        size = pts_file.stat().st_size

        if size < 5_000_000:
            return "too_low"
        elif size > 50_000_000:
            return "too_high"
        else:
            return "good"

    # =====================================================
    # DENSE SWITCH
    # =====================================================
    def _run_dense(self):
        backend = self.config["pipeline"].get("dense_backend", "colmap")

        if backend == "colmap":
            self._run_dense_colmap()

        elif backend == "openmvs":
            self._run_dense_openmvs()

        else:
            raise ValueError(f"Unknown dense backend: {backend}")

    # -------------------------
    # COLMAP DENSE
    # -------------------------
    def _run_dense_colmap(self):
        undistort(self.paths, self.config, self.logger, self.tool_runner)

        try:
            patch_match(self.paths, self.config, self.logger, self.tool_runner)
            stereo_fusion(self.paths, self.config, self.logger, self.tool_runner)
        except Exception as e:
            self.logger.warning(f"Dense failed → {e}")
            raise

    # -------------------------
    # OPENMVS DENSE
    # -------------------------
    def _run_dense_openmvs(self):
        openmvs_export(self.paths, self.config, self.logger, self.tool_runner)
        openmvs_densify(self.paths, self.config, self.logger, self.tool_runner)

    # =====================================================
    # MESH
    # =====================================================
    def _run_mesh(self):
        backend = self.config["pipeline"].get("mesh_backend", "colmap")

        if backend == "openmvs":
            openmvs_mesh(self.paths, self.config, self.logger, self.tool_runner)
        else:
            mesh_reconstruction(self.paths, self.config, self.logger)

    # =====================================================
    # TEXTURE
    # =====================================================
    def _run_texture(self):
        backend = self.config["pipeline"].get("texture_backend", "colmap")

        if backend == "openmvs":
            openmvs_texture(self.paths, self.config, self.logger, self.tool_runner)
        else:
            self.logger.info("Texture skipped (no backend)")

    # =====================================================
    # QUALITY CONTROL
    # =====================================================
    def _force_high_quality_start(self):
        self.config.setdefault("sift", {})["max_num_features"] = 16000

    def _increase_quality(self):
        self.config["sift"]["max_num_features"] += 1000

    def _decrease_quality(self):
        self.config["sift"]["max_num_features"] -= 1000