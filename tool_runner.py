#!/usr/bin/env python3
"""
tool_runner.py

MARK-2 Tool Execution Authority (Canonical)
------------------------------------------
- Sole executor for all external tools
- GPU controlled via environment (COLMAP-correct)
- No hardcoded subcommand flags
- Deterministic, logged, dry-run capable
"""

import subprocess
import time
import os
from pathlib import Path
from typing import List, Optional
from shutil import which


class ToolExecutionError(RuntimeError):
    pass


class ToolRunner:
    """
    Centralized executor for all external tools.
    """

    def __init__(self, config: dict, logger):
        self.config = config
        self.logger = logger
        self.exec_cfg = config["execution"]
        self.tools_cfg = config["tools"]

        self._validate_tools()

    # --------------------------------------------------
    # Core runner
    # --------------------------------------------------

    def run(
        self,
        tool: str,
        args: List[str],
        cwd: Optional[Path] = None,
        check: bool = True,
    ):
        exe = self._resolve_executable(tool)
        cmd = [exe] + [str(a) for a in args]

        env = os.environ.copy()

        # --------------------------------------------------
        # GPU policy (CORRECT for COLMAP & OpenMVS)
        # --------------------------------------------------
        if not self.exec_cfg.get("use_gpu", True):
            env["CUDA_VISIBLE_DEVICES"] = ""
            self.logger.debug("[tool] GPU disabled via CUDA_VISIBLE_DEVICES")
        else:
            # Respect user environment; do not inject flags
            env.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", ""))

        cmd_str = " ".join(cmd)
        self.logger.info(f"[tool:{tool}] CMD: {cmd_str}")

        if self.exec_cfg.get("dry_run", False):
            self.logger.info(f"[tool:{tool}] DRY RUN — skipped")
            return None

        start = time.time()
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(cwd) if cwd else None,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=check,
            )
        except subprocess.CalledProcessError as e:
            self.logger.error(f"[tool:{tool}] FAILED")
            if e.stdout:
                self.logger.error(e.stdout)
            raise ToolExecutionError(cmd_str) from e

        elapsed = time.time() - start
        self.logger.info(f"[tool:{tool}] DONE in {elapsed:.2f}s")

        if proc.stdout and proc.stdout.strip():
            self.logger.debug(proc.stdout)

        return proc

    # --------------------------------------------------
    # Executable resolution
    # --------------------------------------------------

    def _resolve_executable(self, tool: str) -> str:
        tool = tool.lower()

        if tool in self.tools_cfg:
            entry = self.tools_cfg[tool]
            exe = entry["executable"] if isinstance(entry, dict) else entry
        elif "openmvs" in self.tools_cfg and tool in self.tools_cfg["openmvs"]:
            exe = self.tools_cfg["openmvs"][tool]
        else:
            raise KeyError(f"Tool '{tool}' not defined in config")

        exe = str(exe)
        p = Path(exe)

        if p.is_absolute():
            if not p.exists():
                raise FileNotFoundError(f"Executable not found: {exe}")
        else:
            if which(exe) is None:
                raise FileNotFoundError(f"Executable not on PATH: {exe}")

        return exe

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

    def _validate_tools(self) -> None:
        for entry in self.tools_cfg.values():
            if isinstance(entry, dict):
                for exe in entry.values():
                    self._validate_exe(exe)
            else:
                self._validate_exe(entry)

        self.logger.info("[tool] All configured executables validated")

    def _validate_exe(self, exe: str) -> None:
        exe = str(exe)
        p = Path(exe)
        if p.is_absolute():
            if not p.exists():
                raise FileNotFoundError(f"Executable not found: {exe}")
        else:
            if which(exe) is None:
                raise FileNotFoundError(f"Executable not on PATH: {exe}")
