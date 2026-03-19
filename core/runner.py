from utils.paths import ProjectPaths
from utils.logger import setup_logger
from core.tool_runner import ToolRunner

# --- ingestion ---
from stages.ingestion.ingest_images import run as ingest_images
from stages.ingestion.validate_images import run as validate_images
from stages.ingestion.downsample import run as downsample_images

# ----- sparse -----
from stages.colmap_code.feature_extraction import run as feature_extraction
from stages.colmap_code.feature_matching import run as matching

# --- sparse (prepare for next step) ---
# from stages.sparse.feature_extraction import run as feature_extraction
# from stages.sparse.matching import run as matching


class PipelineRunner:
    def __init__(self, config):
        project_root = config["paths"]["project_root"]

        self.paths = ProjectPaths(project_root)
        self.logger = setup_logger(self.paths.log_file)
        self.config = config

        self.tool_runner = ToolRunner(self.logger)

    def run(self):
        self.logger.info("========== PIPELINE START ==========")
        self.logger.info(f"Run directory: {self.paths.run_root}")

        try:
            self._run_ingestion()

            # Uncomment when ready
            # self._run_sparse()

        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise

        self.logger.info("========== PIPELINE END ==========")

    # -----------------------------
    # INGESTION
    # -----------------------------
    def _run_ingestion(self):
        self.logger.info("---- INGESTION STAGE ----")

        ingest_images(self.paths, self.config, self.logger)
        validate_images(self.paths, self.config, self.logger)

        if self.config.get("downsampling", {}).get("enabled", True):
            downsample_images(self.paths, self.config, self.logger)

        self.logger.info("---- INGESTION COMPLETE ----")

    # -----------------------------
    # SPARSE (future)
    # -----------------------------
    def _run_sparse(self):
        self.logger.info("---- SPARSE STAGE ----")

        feature_extraction(self.paths, self.config, self.logger, self.tool_runner)
        matching(self.paths, self.config, self.logger, self.tool_runner)

        self.logger.info("---- SPARSE COMPLETE ----")