#!/usr/bin/env python3
"""
tool_runner.py

Central execution layer for external tools.
"""

import subprocess
import time
import os
from pathlib import Path
from shutil import which


class ToolExecutionError(RuntimeError):
    pass


class ToolRunner:

    def __init__(self, config: dict, logger):

        self.logger = logger
        self.exec_cfg = config.get("execution", {})
        self.tools_cfg = config.get("tools", {})

        self._validate_tools()

    # -------------------------------------------------
    # PUBLIC EXECUTION
    # -------------------------------------------------

    def run(self, tool: str, args=None, cwd: Path | None = None):

        args = args or []

        exe = self._resolve_executable(tool)

        cmd = [exe] + [str(a) for a in args]

        env = os.environ.copy()

        if not self.exec_cfg.get("use_gpu", True):
            env["CUDA_VISIBLE_DEVICES"] = ""

        cmd_str = " ".join(cmd)

        self.logger.info(f"[tool:{tool}] {cmd_str}")

        if self.exec_cfg.get("dry_run", False):
            self.logger.info(f"[tool:{tool}] DRY RUN")
            return

        start = time.time()

        try:

            proc = subprocess.Popen(
                cmd,
                cwd=str(cwd) if cwd else None,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )

            for line in proc.stdout:
                if line.strip():
                    self.logger.info(line.rstrip())

            proc.wait()

            if proc.returncode != 0:
                raise subprocess.CalledProcessError(proc.returncode, cmd)

        except subprocess.CalledProcessError as e:

            self.logger.error(f"[tool:{tool}] FAILED")

            raise ToolExecutionError(tool) from e

        elapsed = time.time() - start

        self.logger.info(f"[tool:{tool}] DONE ({elapsed:.2f}s)")

        return proc

    # -------------------------------------------------
    # TOOL EXISTENCE CHECK
    # -------------------------------------------------

    def has_tool(self, tool: str) -> bool:

        try:
            self._resolve_executable(tool)
            return True
        except Exception:
            return False

    # -------------------------------------------------
    # RESOLVE EXECUTABLE
    # -------------------------------------------------

    def _resolve_executable(self, tool: str):

        exe = None

        if "." in tool:

            parent, child = tool.split(".", 1)

            parent_cfg = self.tools_cfg.get(parent)

            if not isinstance(parent_cfg, dict):
                raise KeyError(f"{parent} not defined as tool group")

            exe = parent_cfg.get(child)

            if exe is None:
                raise KeyError(f"{tool} not defined")

        else:

            exe = self.tools_cfg.get(tool)

            if exe is None:
                raise KeyError(f"Tool '{tool}' not defined")

        if isinstance(exe, dict):
            raise ValueError(f"Tool '{tool}' is a group, not executable")

        exe = self._normalize_exe(exe)

        self._validate_exe(exe)

        return exe

    # -------------------------------------------------
    # NORMALIZE EXECUTABLE
    # -------------------------------------------------

    def _normalize_exe(self, exe: str):

        p = Path(exe)

        if os.name == "nt" and not p.suffix:

            exe_exe = exe + ".exe"

            if which(exe_exe):
                return exe_exe

        return exe

    # -------------------------------------------------
    # VALIDATE CONFIGURED TOOLS
    # -------------------------------------------------

    def _validate_tools(self):

        for name, value in self.tools_cfg.items():

            if isinstance(value, dict):

                for sub_name, exe in value.items():
                    exe = self._normalize_exe(exe)
                    self._validate_exe(exe)

            else:

                exe = self._normalize_exe(value)
                self._validate_exe(exe)

        self.logger.info("[tools] executables validated")

    # -------------------------------------------------
    # VALIDATE EXECUTABLE EXISTS
    # -------------------------------------------------

    def _validate_exe(self, exe: str):

        p = Path(exe)

        if p.is_absolute():

            if not p.exists():
                raise FileNotFoundError(exe)

        else:

            if which(exe) is None:
                raise FileNotFoundError(exe)