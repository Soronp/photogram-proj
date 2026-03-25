import subprocess
import time
from pathlib import Path
from typing import List, Union, Optional


class ToolRunner:
    def __init__(self, logger):
        self.logger = logger

    def run(
        self,
        cmd: Union[str, List[str]],
        cwd: Optional[Path] = None,
        env: Optional[dict] = None,
        stage: str = "unknown",
        allow_failure: bool = False,
        timeout: Optional[int] = None,
        quiet: bool = False,
    ):
        """
        Execute external command with logging, streaming, and timing.

        Returns:
            {
                "elapsed": float,
                "returncode": int
            }
        """

        cmd_str = cmd if isinstance(cmd, str) else " ".join(map(str, cmd))

        if not quiet:
            self.logger.info(f"[{stage}] COMMAND:")
            self.logger.info(cmd_str)

        start_time = time.time()

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=str(cwd) if cwd else None,
                env=env,
                text=True,
                bufsize=1,
            )

            # Stream output live
            for line in iter(process.stdout.readline, ""):
                if line:
                    if not quiet:
                        self.logger.info(f"[{stage}] {line.strip()}")

            process.wait(timeout=timeout)

        except subprocess.TimeoutExpired:
            process.kill()
            msg = f"[{stage}] TIMEOUT after {timeout}s"
            self.logger.error(msg)
            if not allow_failure:
                raise RuntimeError(msg)
            return {"elapsed": None, "returncode": -1}

        elapsed = time.time() - start_time

        self.logger.info(f"[{stage}] Finished in {elapsed:.2f}s")

        if process.returncode != 0:
            msg = f"[{stage}] FAILED (code {process.returncode})"
            self.logger.error(msg)

            if not allow_failure:
                raise RuntimeError(msg)

        return {
            "elapsed": elapsed,
            "returncode": process.returncode
        }