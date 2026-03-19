from utils.paths import ProjectPaths
from utils.logger import setup_logger, save_metrics_json
from core.tool_runner import ToolRunner

# -----------------------------
# INGESTION
# -----------------------------
from stages.ingestion.ingest_images import run as ingest_images
from stages.ingestion.validate_images import run as validate_images
from stages.ingestion.downsample import run as downsample_images

# -----------------------------
# SPARSE
# -----------------------------
from stages.sparse.feature_extraction import run as feature_extraction
from stages.sparse.feature_matching import run as matching
from stages.sparse.mapper import run as mapper

# -----------------------------
# ANALYSIS (NEW)
# -----------------------------
from stages.analysis.dataset_analyzer import run as dataset_analyzer
from stages.analysis.feature_stats import run as feature_stats
from stages.analysis.match_stats import run as match_stats
from stages.analysis.parameter_optimizer import run as parameter_optimizer

# -----------------------------
# DENSE
# -----------------------------
from stages.dense.colmap.image_undistorter import run as undistort
from stages.dense.colmap.patch_match_stereo import run as patch_match
from stages.dense.colmap.stereo_fusion import run as stereo_fusion

# -----------------------------
# MESH
# -----------------------------
from stages.mesh.mesh_reconstruction import run as mesh_reconstruction

# -----------------------------
# TEXTURE
# -----------------------------
from stages.texture.texture_mesh import run as texture_mesh


class PipelineRunner:
    def __init__(self, config):
        project_root = config["paths"]["project_root"]

        self.paths = ProjectPaths(project_root)
        self.logger = setup_logger(self.paths.log_file)
        self.config = config

        self.tool_runner = ToolRunner(self.logger)
        self.analysis_results = {}

    # =====================================================
    # MAIN
    # =====================================================
    def run(self):
        self.logger.info("========== PIPELINE START ==========")
        self.logger.info(f"Run directory: {self.paths.run_root}")

        try:
            self._run_ingestion()
            self._run_sparse_pre_analysis()
            self._run_analysis()          # 🔥 NEW
            self._run_sparse_post_analysis()
            self._run_dense()
            self._run_mesh()
            self._run_texture()

        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise

        self.logger.info("========== PIPELINE END ==========")

    # =====================================================
    # INGESTION
    # =====================================================
    def _run_ingestion(self):
        self.logger.info("---- INGESTION STAGE ----")

        ingest_images(self.paths, self.config, self.logger)
        validate_images(self.paths, self.config, self.logger)

        if self.config["downsampling"]["enabled"]:
            downsample_images(self.paths, self.config, self.logger)

        self.logger.info("---- INGESTION COMPLETE ----")

    # =====================================================
    # SPARSE (PRE-ANALYSIS)
    # =====================================================
    def _run_sparse_pre_analysis(self):
        self.logger.info("---- SPARSE (PRE-ANALYSIS) ----")

        feature_extraction(self.paths, self.config, self.logger, self.tool_runner)
        matching(self.paths, self.config, self.logger, self.tool_runner)

        self.logger.info("---- PRE-ANALYSIS COMPLETE ----")

    # =====================================================
    # ANALYSIS + OPTIMIZATION
    # =====================================================
    def _run_analysis(self):
        self.logger.info("---- ANALYSIS STAGE ----")

        ds = dataset_analyzer(self.paths, self.config, self.logger)
        fs = feature_stats(self.paths, self.config, self.logger)
        ms = match_stats(self.paths, self.config, self.logger)

        stats = {**ds, **fs, **ms}
        self.analysis_results = stats

        self.logger.info(f"Analysis Results: {stats}")

        # 🔥 Optimize config dynamically
        self.config = parameter_optimizer(self.config, stats, self.logger)

        # 🔥 Save JSON metrics
        save_metrics_json(self.paths.metrics, stats, self.config)

        self.logger.info("---- ANALYSIS COMPLETE ----")

    # =====================================================
    # SPARSE (POST-ANALYSIS)
    # =====================================================
    def _run_sparse_post_analysis(self):
        self.logger.info("---- SPARSE (FINAL RECONSTRUCTION) ----")

        mapper(self.paths, self.config, self.logger, self.tool_runner)

        self.logger.info("---- SPARSE COMPLETE ----")

    # =====================================================
    # DENSE
    # =====================================================
    def _run_dense(self):
        self.logger.info("---- DENSE STAGE ----")

        undistort(self.paths, self.config, self.logger, self.tool_runner)
        patch_match(self.paths, self.config, self.logger, self.tool_runner)
        stereo_fusion(self.paths, self.config, self.logger, self.tool_runner)

        self.logger.info("---- DENSE COMPLETE ----")

    # =====================================================
    # MESH
    # =====================================================
    def _run_mesh(self):
        self.logger.info("---- MESH STAGE ----")

        mesh_reconstruction(self.paths, self.config, self.logger)

        self.logger.info("---- MESH COMPLETE ----")

    # =====================================================
    # TEXTURE
    # =====================================================
    def _run_texture(self):
        self.logger.info("---- TEXTURE STAGE ----")

        texture_mesh(self.paths, self.config, self.logger, self.tool_runner)

        self.logger.info("---- TEXTURE COMPLETE ----")