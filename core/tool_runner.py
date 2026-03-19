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
        Execute external command with logging + timing.
        """

        cmd_str = cmd if isinstance(cmd, str) else " ".join(cmd)

        self.logger.info(f"[{stage}] COMMAND:")
        self.logger.info(cmd_str)

        start_time = time.time()

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=cwd,
            env=env,
            text=True,
        )

        # Stream output
        for line in iter(process.stdout.readline, ""):
            if line:
                self.logger.info(f"[{stage}] {line.strip()}")

        process.wait()
        elapsed = time.time() - start_time

        self.logger.info(f"[{stage}] Finished in {elapsed:.2f}s")

        if process.returncode != 0:
            msg = f"[{stage}] FAILED (code {process.returncode})"
            self.logger.error(msg)

            if not allow_failure:
                raise RuntimeError(msg)

        return elapsed