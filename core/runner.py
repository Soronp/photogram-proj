from pathlib import Path

from utils.paths import ProjectPaths
from utils.logger import setup_logger
from core.tool_runner import ToolRunner
from core.stage import Stage


class Runner:
    """
    Pipeline runner responsible for executing stages sequentially.
    """

    def __init__(self, project_root, config=None):

        # Initialize paths
        self.paths = ProjectPaths(project_root)
        self.paths.create_all()

        # Initialize logger
        self.logger = setup_logger(self.paths.logs)

        # Initialize tool runner
        self.tool_runner = ToolRunner(self.logger)

        # Config placeholder
        self.config = config if config else {}

        self.logger.info("Runner initialized")
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


# --------------------------------------------------------
# TEST STAGE
# --------------------------------------------------------

class TestStage(Stage):

    name = "test_stage"

    def run(self, paths, config, logger, tool_runner):

        logger.info("TestStage is running")
        print("\n>>> TEST STAGE EXECUTED SUCCESSFULLY <<<\n")

        logger.info("Paths summary:")

        for key, value in paths.summary().items():
            logger.info(f"{key} -> {value}")


# --------------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------------

if __name__ == "__main__":

    project_root = Path(".")  # current folder

    runner = Runner(project_root)

    pipeline = [
        TestStage()
    ]

    runner.run(pipeline)