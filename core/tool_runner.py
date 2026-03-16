import subprocess
import time
from pathlib import Path


class ToolRunner:
    """
    Responsible for executing external photogrammetry tools
    such as COLMAP and OpenMVS in a controlled manner.
    """

    def __init__(self, logger):
        self.logger = logger

    def run(self, command, cwd=None, env=None, check=True):
        """
        Execute an external command.

        Parameters
        ----------
        command : list[str]
            Command and arguments.
        cwd : Path | None
            Working directory.
        env : dict | None
            Environment variables.
        check : bool
            If True, raise error on failure.
        """

        cmd_str = " ".join(map(str, command))

        self.logger.info(f"Running command: {cmd_str}")

        start_time = time.time()

        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        # Stream output live to logger
        for line in process.stdout:
            self.logger.info(line.strip())

        process.wait()

        runtime = time.time() - start_time

        if process.returncode != 0:
            self.logger.error(f"Command failed with exit code {process.returncode}")
            if check:
                raise RuntimeError(f"Command failed: {cmd_str}")

        self.logger.info(f"Command finished in {runtime:.2f} seconds")

        return process.returncode