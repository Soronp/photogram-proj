import subprocess
import time
from pathlib import Path


class ToolRunner:
    def __init__(self, logger):
        self.logger = logger

    def run(
        self,
        cmd,
        cwd: Path = None,
        env: dict = None,
        stage: str = "unknown",
        allow_failure: bool = False,
    ):
        """
        Execute a shell command with logging and error handling.
        """

        if isinstance(cmd, list):
            cmd_str = " ".join(cmd)
        else:
            cmd_str = cmd

        self.logger.info(f"[{stage}] Running command:")
        self.logger.info(cmd_str)

        start_time = time.time()

        process = subprocess.Popen(
            cmd,
            shell=isinstance(cmd, str),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=cwd,
            env=env,
            text=True,
        )

        # Stream output live
        for line in process.stdout:
            self.logger.info(f"[{stage}] {line.strip()}")

        process.wait()
        elapsed = time.time() - start_time

        self.logger.info(f"[{stage}] Finished in {elapsed:.2f}s")

        if process.returncode != 0:
            msg = f"[{stage}] Command failed with code {process.returncode}"
            self.logger.error(msg)

            if not allow_failure:
                raise RuntimeError(msg)

        return elapsed