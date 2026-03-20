from copy import deepcopy

from utils.paths import ProjectPaths
from utils.logger import setup_logger, save_metrics_json
from core.tool_runner import ToolRunner

# INGESTION
from stages.ingestion.ingest_images import run as ingest_images
from stages.ingestion.validate_images import run as validate_images
from stages.ingestion.downsample import run as downsample_images

# SPARSE (PROBE)
from stages.sparse.feature_extraction import run as feature_extraction
from stages.sparse.feature_matching import run as matching

# ANALYSIS
from stages.analysis.dataset_analyzer import run as dataset_analyzer
from stages.analysis.feature_stats import run as feature_stats
from stages.analysis.match_stats import run as match_stats
from stages.analysis.parameter_optimizer import run as parameter_optimizer

# SPARSE FINAL
from stages.sparse.mapper import run as mapper

# DENSE
from stages.dense.colmap.image_undistorter import run as undistort
from stages.dense.colmap.patch_match_stereo import run as patch_match
from stages.dense.colmap.stereo_fusion import run as stereo_fusion

# MESH + TEXTURE
from stages.mesh.mesh_reconstruction import run as mesh_reconstruction
from stages.texture.texture_mesh import run as texture_mesh


class PipelineRunner:

    def __init__(self, config):
        project_root = config["paths"]["project_root"]

        self.paths = ProjectPaths(project_root)
        self.logger = setup_logger(self.paths.log_file)

        self.base_config = deepcopy(config)
        self.tool_runner = ToolRunner(self.logger)

        self.analysis_results = {}

        # 🔥 HARD LIMIT: max correction passes
        self.max_adjustments = 2

    # =====================================================
    # MAIN EXECUTION
    # =====================================================
    def run(self):
        self.logger.info("========== PIPELINE START ==========")

        # -----------------------------
        # STEP 1: INGESTION
        # -----------------------------
        self.config = deepcopy(self.base_config)
        self._run_ingestion()

        # -----------------------------
        # STEP 2: PROBE
        # -----------------------------
        self._run_probe_sparse()

        # -----------------------------
        # STEP 3: ANALYSIS + INITIAL CONFIG
        # -----------------------------
        self._run_analysis()

        # 🔥 START HIGH STRATEGY
        self._force_high_quality_start()

        # =====================================================
        # 🔥 ADAPTIVE LOOP (NOT RETRIES)
        # =====================================================
        for adjustment in range(self.max_adjustments + 1):

            self.logger.info(f"===== ADAPTIVE PASS {adjustment + 1} =====")

            # -----------------------------
            # SPARSE + DENSE
            # -----------------------------
            self._run_final_sparse()
            self._run_dense()

            # -----------------------------
            # EVALUATION
            # -----------------------------
            status = self._evaluate_quality()

            if status == "good":
                self.logger.info("✅ Optimal reconstruction achieved")
                break

            elif status == "too_low":
                self.logger.warning("⬆️ Too sparse → increasing quality")
                self._increase_quality()

            elif status == "too_high":
                self.logger.warning("⬇️ Too dense → reducing load")
                self._decrease_quality()

            if adjustment == self.max_adjustments:
                self.logger.warning("Reached adjustment limit")

        # -----------------------------
        # FINAL STAGES
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

        if self.config.get("downsampling", {}).get("enabled", True):
            downsample_images(self.paths, self.config, self.logger)

    # =====================================================
    def _run_probe_sparse(self):
        feature_extraction(self.paths, self.config, self.logger, self.tool_runner)
        matching(self.paths, self.config, self.logger, self.tool_runner)

    # =====================================================
    def _run_analysis(self):
        ds = dataset_analyzer(self.paths, self.config, self.logger)
        fs = feature_stats(self.paths, self.config, self.logger)
        ms = match_stats(self.paths, self.config, self.logger)

        self.analysis_results = {
            "dataset": ds,
            "features": fs,
            "matches": ms
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
        for section, values in new_params.items():
            self.config.setdefault(section, {})
            self.config[section].update(values)

    # =====================================================
    def _run_final_sparse(self):
        mapper(self.paths, self.config, self.logger, self.tool_runner)

    def _run_dense(self):
        undistort(self.paths, self.config, self.logger, self.tool_runner)
        patch_match(self.paths, self.config, self.logger, self.tool_runner)
        stereo_fusion(self.paths, self.config, self.logger, self.tool_runner)

    # =====================================================
    # 🔥 QUALITY EVALUATION (CRITICAL)
    # =====================================================
    def _evaluate_quality(self):
        try:
            import open3d as o3d

            fused_path = self.paths.dense / "fused.ply"
            if not fused_path.exists():
                return "too_low"

            pcd = o3d.io.read_point_cloud(str(fused_path))
            num_points = len(pcd.points)

            self.logger.info(f"Quality check: {num_points} points")

            # 🔥 tuned for your machine
            if num_points < 200000:
                return "too_low"

            elif num_points > 800000:
                return "too_high"

            else:
                return "good"

        except Exception:
            return "too_low"

    # =====================================================
    # 🔥 START HIGH
    # =====================================================
    def _force_high_quality_start(self):
        self.logger.info("Forcing HIGH QUALITY starting config")

        self.config["sift"]["max_num_features"] = 16000

        self.config["dense"].update({
            "window_radius": 9,
            "num_samples": 30,
            "num_iterations": 7
        })

        self.config["fusion"]["min_num_pixels"] = 2

    # =====================================================
    # 🔥 ADAPTATION LOGIC
    # =====================================================
    def _increase_quality(self):
        self.config["sift"]["max_num_features"] += 2000

        self.config["dense"]["num_samples"] += 5
        self.config["dense"]["window_radius"] = min(
            self.config["dense"]["window_radius"] + 1, 9
        )

        self.config["fusion"]["min_num_pixels"] = max(
            2, self.config["fusion"]["min_num_pixels"] - 1
        )

    def _decrease_quality(self):
        self.config["sift"]["max_num_features"] -= 2000

        self.config["dense"]["num_samples"] -= 5
        self.config["dense"]["window_radius"] = max(
            3, self.config["dense"]["window_radius"] - 1
        )

        self.config["fusion"]["min_num_pixels"] += 1

    # =====================================================
    def _run_mesh(self):
        mesh_reconstruction(self.paths, self.config, self.logger)

    def _run_texture(self):
        texture_mesh(self.paths, self.config, self.logger, self.tool_runner)