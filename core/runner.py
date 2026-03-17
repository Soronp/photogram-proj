import sys
from pathlib import Path

# ------------------------------------------------------------------
# Fix import path so project modules can be found
# ------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

# ------------------------------------------------------------------

from utils.paths import ProjectPaths
from utils.logger import setup_logger
from core.tool_runner import ToolRunner
from core.stage import Stage
from stages.preprocessing.image_downsample import ImageDownsampleStage
from stages.colmap_code.feature_extraction import FeatureExtractionStage
from stages.colmap_code.feature_matching import FeatureMatchingStage
from stages.colmap_code.mapper import MapperStage
from stages.colmap_code.image_undistorter import ImageUndistorterStage
from stages.colmap_code.patch_match_stereo import PatchMatchStereoStage
from stages.colmap_code.stereo_fusion import StereoFusionStage

class Runner:
    """
    Pipeline runner responsible for executing stages sequentially.
    """

    def __init__(self, dataset_path, output_root, config=None):

        self.dataset_path = Path(dataset_path).resolve()
        self.output_root = Path(output_root).resolve()

        # Initialize paths
        self.paths = ProjectPaths(self.output_root)
        self.paths.images = self.dataset_path
        self.paths.create_all()

        # Initialize logger
        self.logger = setup_logger(self.paths.logs)

        # Initialize tool runner
        self.tool_runner = ToolRunner(self.logger)

        self.config = config if config else {}

        self.logger.info("Runner initialized")
        self.logger.info(f"Dataset: {self.dataset_path}")
        self.logger.info(f"Run directory: {self.paths.run_root}")

    def run(self, stages):

        self.logger.info("Pipeline starting")

        for stage in stages:

            self.logger.info(f"Starting stage: {stage.name}")

            try:

                stage.run(
                    paths=self.paths,
                    config=self.config,
                    logger=self.logger,
                    tool_runner=self.tool_runner
                )

                self.logger.info(f"Stage completed: {stage.name}")

            except Exception as e:

                self.logger.error(f"Stage failed: {stage.name}")
                self.logger.error(str(e))

                raise

        self.logger.info("Pipeline finished successfully")


# ------------------------------------------------------------------
# TEST STAGE
# ------------------------------------------------------------------

class TestStage(Stage):

    name = "test_stage"

    def run(self, paths, config, logger, tool_runner):

        logger.info("TestStage running")

        print("\n===== TEST STAGE EXECUTED SUCCESSFULLY =====\n")

        logger.info("Paths summary:")

        for key, value in paths.summary().items():
            logger.info(f"{key} -> {value}")


# ------------------------------------------------------------------
# MAIN PROGRAM
# ------------------------------------------------------------------

def get_user_paths():

    print("\n--- Photogrammetry Pipeline ---\n")

    dataset = input("Enter dataset image folder path: ").strip()
    output = input("Enter output project directory: ").strip()

    dataset = Path(dataset)
    output = Path(output)

    if not dataset.exists():
        raise ValueError("Dataset path does not exist")

    return dataset, output


if __name__ == "__main__":

    dataset_path, output_root = get_user_paths()

    runner = Runner(dataset_path, output_root)

    pipeline = [
        ImageDownsampleStage(),
        FeatureExtractionStage(),
        FeatureMatchingStage(),
        MapperStage(),
        ImageUndistorterStage(),
        PatchMatchStereoStage(),
        StereoFusionStage(),
    ]

    runner.run(pipeline)