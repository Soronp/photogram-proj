from copy import deepcopy

from utils.paths import ProjectPaths
from utils.logger import setup_logger, save_metrics_json
from core.tool_runner import ToolRunner

# INGESTION
from stages.ingestion.ingest_images import run as ingest_images
from stages.ingestion.validate_images import run as validate_images
from stages.ingestion.downsample import run as downsample_images

# SPARSE
from stages.sparse.feature_extraction import run as feature_extraction
from stages.sparse.feature_matching import run as matching
from stages.sparse.mapper import run as mapper

# ANALYSIS
from stages.analysis.dataset_analyzer import run as dataset_analyzer
from stages.analysis.feature_stats import run as feature_stats
from stages.analysis.match_stats import run as match_stats
from stages.analysis.parameter_optimizer import run as parameter_optimizer

# DENSE (COLMAP)
from stages.dense.colmap.image_undistorter import run as undistort
from stages.dense.colmap.patch_match_stereo import run as patch_match
from stages.dense.colmap.stereo_fusion import run as stereo_fusion

# FINAL
from stages.mesh.mesh_reconstruction import run as mesh_reconstruction
from stages.texture.texture_mesh import run as texture_mesh


class PipelineRunner:

    def __init__(self, config, pipeline_type="A"):
        """
        pipeline_type:
            A → COLMAP full pipeline
            B → GLOMAP sparse + COLMAP dense
        """

        project_root = config["paths"]["project_root"]

        self.paths = ProjectPaths(project_root)
        self.logger = setup_logger(self.paths.log_file)

        self.base_config = deepcopy(config)
        self.config = deepcopy(config)

        self.pipeline_type = pipeline_type

        self.tool_runner = ToolRunner(self.logger)
        self.analysis_results = {}

        self.max_adjustments = 2

        # 🔥 Inject backend based on pipeline
        self._configure_pipeline()

    # =====================================================
    # PIPELINE CONFIGURATION
    # =====================================================
    def _configure_pipeline(self):

        if self.pipeline_type == "A":
            self.logger.info("Pipeline A: COLMAP full pipeline")
            self.config["pipeline"]["sparse_backend"] = "colmap"

        elif self.pipeline_type == "B":
            self.logger.info("Pipeline B: GLOMAP + COLMAP dense")
            self.config["pipeline"]["sparse_backend"] = "glomap"

        else:
            raise ValueError(f"Unknown pipeline type: {self.pipeline_type}")

    # =====================================================
    # MAIN
    # =====================================================
    def run(self):

        self.logger.info("========== PIPELINE START ==========")

        # -----------------------------
        # INGESTION
        # -----------------------------
        self._run_ingestion()

        # -----------------------------
        # FEATURES (ONCE)
        # -----------------------------
        self._run_feature_pipeline()

        # -----------------------------
        # ANALYSIS
        # -----------------------------
        self._run_analysis()

        self.config["analysis_results"] = self.analysis_results
        self._force_high_quality_start()

        # =====================================================
        # REFINEMENT LOOP
        # =====================================================
        for i in range(self.max_adjustments + 1):

            self.logger.info(f"===== REFINEMENT PASS {i + 1} =====")

            # -----------------------------
            # SPARSE (backend controlled)
            # -----------------------------
            self._run_mapper()

            # -----------------------------
            # DENSE (still COLMAP)
            # -----------------------------
            self._run_dense_colmap()

            # -----------------------------
            # QUALITY CHECK
            # -----------------------------
            status = self._evaluate_quality()

            if status == "good":
                self.logger.info("✅ Optimal reconstruction achieved")
                break

            elif status == "too_low":
                self.logger.warning("⬆️ Increasing quality")
                self._increase_quality()

            elif status == "too_high":
                self.logger.warning("⬇️ Reducing load")
                self._decrease_quality()

        # -----------------------------
        # FINAL
        # -----------------------------
        self._run_mesh()
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
    # FEATURE PIPELINE
    # =====================================================
    def _run_feature_pipeline(self):
        self.logger.info("Running feature pipeline (ONE-TIME)")

        feature_extraction(self.paths, self.config, self.logger, self.tool_runner)
        matching(self.paths, self.config, self.logger, self.tool_runner)

    # =====================================================
    # ANALYSIS
    # =====================================================
    def _run_analysis(self):

        ds = dataset_analyzer(self.paths, self.config, self.logger)
        fs = feature_stats(self.paths, self.config, self.logger)
        ms = match_stats(self.paths, self.config, self.logger)

        self.analysis_results = {
            "dataset": ds or {},
            "features": fs or {},
            "matches": ms or {}
        }

        self.logger.info(f"Analysis Summary: {self.analysis_results}")

        optimized = parameter_optimizer(
            self.config,
            self.analysis_results,
            self.logger
        )

        self._update_config(optimized)

        save_metrics_json(
            self.paths.metrics,
            self.analysis_results,
            self.config,
            logger=self.logger
        )

    # =====================================================
    def _update_config(self, new_params):
        for k, v in new_params.items():
            self.config.setdefault(k, {})
            self.config[k].update(v)

    # =====================================================
    # SPARSE
    # =====================================================
    def _run_mapper(self):
        mapper(self.paths, self.config, self.logger, self.tool_runner)

    # =====================================================
    # DENSE (COLMAP ONLY FOR NOW)
    # =====================================================
    def _run_dense_colmap(self):

        undistort(self.paths, self.config, self.logger, self.tool_runner)

        try:
            self.logger.info("Attempting GPU dense reconstruction...")
            patch_match(self.paths, {**self.config, "dense": {"use_gpu": True}}, self.logger, self.tool_runner)
            stereo_fusion(self.paths, {**self.config, "fusion": {"use_gpu": True}}, self.logger, self.tool_runner)
            self.logger.info("✅ GPU dense reconstruction successful")

        except Exception as e:
            self.logger.warning(f"GPU failed: {e}")
            self.logger.info("Falling back to CPU...")

            patch_match(self.paths, {**self.config, "dense": {"use_gpu": False}}, self.logger, self.tool_runner)
            stereo_fusion(self.paths, {**self.config, "fusion": {"use_gpu": False}}, self.logger, self.tool_runner)

            self.logger.info("✅ CPU dense reconstruction complete")

    # =====================================================
    # QUALITY
    # =====================================================
    def _evaluate_quality(self):

        try:
            import open3d as o3d

            fused = self.paths.dense / "fused.ply"

            if not fused.exists():
                return "too_low"

            pcd = o3d.io.read_point_cloud(str(fused))
            pts = len(pcd.points)

            self.logger.info(f"Quality: {pts} points")

            if pts < 200000:
                return "too_low"
            elif pts > 900000:
                return "too_high"
            else:
                return "good"

        except Exception as e:
            self.logger.warning(f"Quality check failed: {e}")
            return "too_low"

    # =====================================================
    def _force_high_quality_start(self):
        self.config.setdefault("sift", {})
        self.config.setdefault("dense", {})
        self.config.setdefault("fusion", {})

        self.config["sift"]["max_num_features"] = max(
            16000,
            self.config["sift"].get("max_num_features", 16000)
        )

        self.config["dense"].update({
            "window_radius": 9,
            "num_samples": 30,
            "num_iterations": 7,
            "use_gpu": True
        })

        self.config["fusion"]["min_num_pixels"] = 2
        self.config["fusion"]["use_gpu"] = True

    # =====================================================
    def _increase_quality(self):
        self.config["sift"]["max_num_features"] += 2000
        self.config["dense"]["num_samples"] += 5
        self.config["fusion"]["min_num_pixels"] = max(
            2, self.config["fusion"]["min_num_pixels"] - 1
        )

    def _decrease_quality(self):
        self.config["sift"]["max_num_features"] -= 2000
        self.config["dense"]["num_samples"] -= 5
        self.config["fusion"]["min_num_pixels"] += 1

    # =====================================================
    def _run_mesh(self):
        mesh_reconstruction(self.paths, self.config, self.logger)

    def _run_texture(self):
        texture_mesh(self.paths, self.config, self.logger, self.tool_runner)