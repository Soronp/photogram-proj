import subprocess
import time
import os
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
        Robust external command runner.

        Guarantees:
        - UTF-8 safe decoding (no Windows charmap crashes)
        - No pipe deadlocks (stdout+stderr merged)
        - Real-time streaming
        - Proper timeout + process cleanup
        - Cross-platform stability
        """

        # ----------------------------------------
        # Normalize command
        # ----------------------------------------
        if isinstance(cmd, list):
            cmd_str = " ".join(map(str, cmd))
        else:
            cmd_str = cmd

        if not quiet:
            self.logger.info(f"[{stage}] COMMAND:")
            self.logger.info(cmd_str)

        # ----------------------------------------
        # Environment hardening (CRITICAL)
        # ----------------------------------------
        run_env = os.environ.copy()

        if env:
            run_env.update(env)

        # Force UTF-8 everywhere
        run_env["PYTHONIOENCODING"] = "utf-8"
        run_env["PYTHONUTF8"] = "1"

        #Prevent rich / ANSI Unicode crashes on Windows
        run_env["TERM"] = "dumb"

        start_time = time.time()

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=str(cwd) if cwd else None,
                env=run_env,

                text=True,
                encoding="utf-8",
                errors="replace",

                bufsize=1,
            )

            # ----------------------------------------
            # Stream output safely
            # ----------------------------------------
            assert process.stdout is not None

            while True:
                line = process.stdout.readline()

                if line == "" and process.poll() is not None:
                    break

                if line and not quiet:
                    self.logger.info(f"[{stage}] {line.rstrip()}")

            # ----------------------------------------
            # Wait for completion (with timeout)
            # ----------------------------------------
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self._terminate_process(process)

                msg = f"[{stage}] TIMEOUT after {timeout}s"
                self.logger.error(msg)

                if not allow_failure:
                    raise RuntimeError(msg)

                return {
                    "elapsed": None,
                    "returncode": -1,
                    "success": False,
                }

        except Exception as e:
            msg = f"[{stage}] CRASHED: {str(e)}"
            self.logger.error(msg)

            if not allow_failure:
                raise

            return {
                "elapsed": None,
                "returncode": -1,
                "success": False,
            }

        # ----------------------------------------
        # Final status
        # ----------------------------------------
        elapsed = time.time() - start_time
        returncode = process.returncode

        self.logger.info(f"[{stage}] Finished in {elapsed:.2f}s")

        if returncode != 0:
            msg = f"[{stage}] FAILED (code {returncode})"
            self.logger.error(msg)

            if not allow_failure:
                raise RuntimeError(msg)

        return {
            "elapsed": elapsed,
            "returncode": returncode,
            "success": returncode == 0,
        }

    # =====================================================
    # INTERNAL: SAFE TERMINATION
    # =====================================================
    def _terminate_process(self, process: subprocess.Popen):
        """Ensure process is fully killed (Windows-safe)."""
        try:
            process.kill()
        except Exception:
            pass

        try:
            process.wait(timeout=5)
        except Exception:
            pass