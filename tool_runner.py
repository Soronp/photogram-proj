#!/usr/bin/env python3
"""
tool_runner.py

MARK-2 Tool Execution Authority (Fixed)
---------------------------------------
- Pulls executables from utils/config.py
- Supports OpenMVS aliases
- Absolute paths bypass PATH check
- GPU/CPU selection per tool
- Centralized logging and dry-run
"""

import subprocess
import time
from pathlib import Path
from typing import List, Dict, Optional
from shutil import which

from utils.config import COLMAP_EXE, GLOMAP_EXE, OPENMVS_TOOLS, validate_executables


class ToolExecutionError(RuntimeError):
    pass


class ToolRunner:
    """
    Centralized tool executor for MARK-2 pipelines.
    """

    GPU_FLAGS = {
        "feature_extractor": [],  # COLMAP CPU-only
        "exhaustive_matcher": ["--SiftMatching.use_gpu", "1"],
        "sequential_matcher": ["--SiftMatching.use_gpu", "1"],
        "mapper": ["--Mapper.use_gpu", "1"],
        "patch_match_stereo": ["--PatchMatchStereo.use_gpu", "1"],
        "stereo_fusion": ["--StereoFusion.use_gpu", "1"],
    }

    DEFAULT_EXECUTABLES = {
        "colmap": COLMAP_EXE,
        "glomap": GLOMAP_EXE,
        "interface_colmap": OPENMVS_TOOLS["interface_colmap"],
        "densify": OPENMVS_TOOLS["densify"],
        "mesh": OPENMVS_TOOLS["mesh"],
        "texture": OPENMVS_TOOLS["texture"],
    }

    ALIASES = {
        "openmvs": "densify",
    }

    def __init__(self, config: dict, logger):
        self.config = config
        self.logger = logger
        self.exec_cfg = config.get("execution", {})

        # Validate all tools that are simple command names (not absolute paths)
        validate_executables()

    # -------------------------
    # Core runner
    # -------------------------
    def run(
        self,
        tool: str,
        args: List[str],
        cwd: Optional[Path] = None,
        env: Optional[Dict[str, str]] = None,
        check: bool = True,
    ):
        exe_path = self._resolve_executable(tool)
        cmd = [exe_path] + args

        # Append GPU flags if enabled
        if self.exec_cfg.get("use_gpu", True):
            gpu_flags = self.GPU_FLAGS.get(tool.lower(), [])
            cmd += gpu_flags

        cmd_str = " ".join(map(str, cmd))
        self.logger.info(f"[tool:{tool}] CMD: {cmd_str}")

        if self.exec_cfg.get("dry_run", False):
            self.logger.info(f"[tool:{tool}] DRY RUN — skipped")
            return

        start = time.time()
        try:
            proc = subprocess.run(
                cmd,
                cwd=cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=check,
            )
        except subprocess.CalledProcessError as e:
            self.logger.error(f"[tool:{tool}] FAILED")
            self.logger.error(e.stdout)
            raise ToolExecutionError(cmd_str) from e

        elapsed = time.time() - start
        self.logger.info(f"[tool:{tool}] DONE in {elapsed:.2f}s")

        if proc.stdout.strip():
            self.logger.debug(proc.stdout)

        return proc

    # -------------------------
    # Tool resolution
    # -------------------------
    def _resolve_executable(self, tool: str) -> str:
        """
        Resolve executable for a given tool.
        Absolute paths are used directly.
        """
        tool_key = self.ALIASES.get(tool.lower(), tool.lower())

        # Check config overrides first
        tools_config = self.config.get("tools", {})
        if tool_key in tools_config:
            entry = tools_config[tool_key]
            exe_path = entry.get("executable") if isinstance(entry, dict) else entry
        else:
            exe_path = self.DEFAULT_EXECUTABLES.get(tool_key, tool_key)

        exe_path = str(exe_path)  # ensure string
        exe_path_obj = Path(exe_path)

        # If relative (not absolute), check PATH
        if not exe_path_obj.is_absolute():
            if which(exe_path) is None:
                raise FileNotFoundError(f"Executable for {tool} not found on PATH: {exe_path}")
        # If absolute, just make sure the file exists
        else:
            if not exe_path_obj.exists():
                raise FileNotFoundError(f"Executable for {tool} not found at: {exe_path}")

        return exe_path
