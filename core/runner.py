from copy import deepcopy

from utils.paths import ProjectPaths
from utils.logger import setup_logger, save_metrics_json
from core.tool_runner import ToolRunner

# STAGES
from stages.ingestion.ingest_images import run as ingest_images
from stages.ingestion.validate_images import run as validate_images
from stages.ingestion.downsample import run as downsample_images

from stages.sparse.feature_extraction import run as feature_extraction
from stages.sparse.feature_matching import run as matching
from stages.sparse.mapper import run as mapper

from stages.analysis.dataset_analyzer import run as dataset_analyzer
from stages.analysis.feature_stats import run as feature_stats
from stages.analysis.match_stats import run as match_stats
from stages.analysis.parameter_optimizer import run as parameter_optimizer

from stages.dense.colmap.image_undistorter import run as undistort
from stages.dense.colmap.patch_match_stereo import run as patch_match
from stages.dense.colmap.stereo_fusion import run as stereo_fusion

from stages.mesh.mesh_reconstruction import run as mesh_reconstruction
from stages.texture.texture_mesh import run as texture_mesh


class PipelineRunner:

    def __init__(self, config, pipeline_type="A"):

        project_root = config["paths"]["project_root"]

        self.paths = ProjectPaths(project_root)
        self.logger = setup_logger(self.paths.log_file)

        self.base_config = deepcopy(config)
        self.config = deepcopy(config)

        self.pipeline_type = pipeline_type
        self.tool_runner = ToolRunner(self.logger)

        self.analysis_results = {}
        self.max_adjustments = 2

        self._configure_pipeline()

    # =====================================================
    # PIPELINE MODE
    # =====================================================
    def _configure_pipeline(self):

        if self.pipeline_type == "A":
            self.logger.info("Pipeline A: COLMAP full")
            self.config["sparse"]["backend"] = "colmap"

        elif self.pipeline_type == "B":
            self.logger.info("Pipeline B: GLOMAP sparse + COLMAP dense")
            self.config["sparse"]["backend"] = "glomap"

        else:
            raise ValueError("Invalid pipeline type")

    # =====================================================
    # MAIN
    # =====================================================
    def run(self):

        self.logger.info("========== PIPELINE START ==========")

        self._run_ingestion()
        self._run_feature_pipeline()
        self._run_analysis()

        self.config["analysis_results"] = self.analysis_results
        self._force_high_quality_start()

        # =====================================================
        # 🔥 SPARSE REFINEMENT ONLY
        # =====================================================
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

        # =====================================================
        # 🔥 SINGLE DENSE RUN (CRITICAL FIX)
        # =====================================================
        self.logger.info("===== FINAL DENSE RECONSTRUCTION =====")
        self._run_dense_once()

        # =====================================================
        # FINAL
        # =====================================================
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
    # FEATURES
    # =====================================================
    def _run_feature_pipeline(self):
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

    def _update_config(self, new_params):
        for k, v in new_params.items():
            self.config.setdefault(k, {})
            self.config[k].update(v)

    # =====================================================
    # SPARSE
    # =====================================================
    def _run_mapper(self):
        mapper(self.paths, self.config, self.logger, self.tool_runner)

    def _evaluate_sparse(self):

        model = self.paths.sparse_model
        pts_file = model / "points3D.bin"

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
    # 🔥 DENSE ONCE
    # =====================================================
    def _run_dense_once(self):

        undistort(self.paths, self.config, self.logger, self.tool_runner)

        try:
            patch_match(self.paths, self.config, self.logger, self.tool_runner)
            stereo_fusion(self.paths, self.config, self.logger, self.tool_runner)

        except Exception:
            self.logger.warning("GPU failed → switching to CPU")

            cfg = deepcopy(self.config)
            cfg["dense"]["use_gpu"] = False

            patch_match(self.paths, cfg, self.logger, self.tool_runner)
            stereo_fusion(self.paths, cfg, self.logger, self.tool_runner)

    # =====================================================
    # QUALITY CONTROL BASELINE
    # =====================================================
    def _force_high_quality_start(self):

        self.config.setdefault("sift", {})
        self.config.setdefault("dense", {})
        self.config.setdefault("fusion", {})

        self.config["sift"]["max_num_features"] = 16000

        # 🔥 IMPORTANT: NOT OVERKILL
        self.config["dense"].update({
            "window_radius": 7,
            "num_samples": 20,
            "num_iterations": 5,
            "use_gpu": True
        })

        self.config["fusion"]["min_num_pixels"] = 3

    # =====================================================
    def _increase_quality(self):
        self.config["sift"]["max_num_features"] += 1000

    def _decrease_quality(self):
        self.config["sift"]["max_num_features"] -= 1000

    # =====================================================
    def _run_mesh(self):
        mesh_reconstruction(self.paths, self.config, self.logger)

    def _run_texture(self):
        texture_mesh(self.paths, self.config, self.logger, self.tool_runner)