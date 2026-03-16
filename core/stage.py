from abc import ABC, abstractmethod


class Stage(ABC):
    """
    Abstract base class for pipeline stages.

    Every stage must implement the run() method.
    """

    name = "base_stage"

    def __init__(self):
        pass

    @abstractmethod
    def run(self, paths, config, logger, tool_runner):
        """
        Execute the stage.

        Parameters
        ----------
        paths : ProjectPaths
            Centralized project paths.
        config : dict
            Pipeline configuration.
        logger : logging.Logger
            Pipeline logger.
        tool_runner : ToolRunner
            External tool executor.
        """
        pass