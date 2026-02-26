#!/usr/bin/env python3
"""
RunManager (Deterministic + Strict Single Active Run)

Responsibilities:
- Discover runs
- Create new run
- Resume specific run
- Resume latest run
- Maintain single active run
"""

from pathlib import Path
from typing import Optional, List
from runs.run_context import RunContext
from utils.paths import ProjectPaths


class RunManager:

    def __init__(self, project_paths: ProjectPaths):
        self.paths = project_paths
        self.paths.runs.mkdir(parents=True, exist_ok=True)
        self.active_run: Optional[RunContext] = None

    # --------------------------------------------------
    # Discovery
    # --------------------------------------------------

    def list_runs(self) -> List[str]:
        runs = [
            p.name for p in self.paths.runs.iterdir()
            if p.is_dir() and p.name.startswith("run_")
        ]
        return sorted(runs)

    def latest_run_id(self) -> Optional[str]:
        runs = self.list_runs()
        return runs[-1] if runs else None

    # --------------------------------------------------
    # Creation
    # --------------------------------------------------

    def start_new_run(
        self,
        project_root: Path,
        input_path: Path,
        output_path: Optional[Path] = None
    ) -> RunContext:

        if self.active_run is not None:
            raise RuntimeError("A run is already active")

        run_ctx = RunContext(self.paths.runs)
        run_ctx.initialize(project_root, input_path)

        self.active_run = run_ctx
        self._maybe_symlink(output_path)

        return run_ctx

    # --------------------------------------------------
    # Resume
    # --------------------------------------------------

    def resume_run(
        self,
        run_id: str,
        output_path: Optional[Path] = None
    ) -> RunContext:

        if self.active_run is not None:
            raise RuntimeError("A run is already active")

        run_ctx = RunContext(self.paths.runs, run_id=run_id)
        run_ctx.validate()

        self.active_run = run_ctx
        self._maybe_symlink(output_path)

        return run_ctx

    def resume_latest(
        self,
        output_path: Optional[Path] = None
    ) -> Optional[RunContext]:

        latest = self.latest_run_id()
        if latest is None:
            return None

        return self.resume_run(latest, output_path)

    # --------------------------------------------------
    # Finalization
    # --------------------------------------------------

    def finish_run(self, success: bool):
        if self.active_run:
            self.active_run.finalize(success)
            self.active_run = None

    # --------------------------------------------------
    # Utilities
    # --------------------------------------------------

    def _maybe_symlink(self, output_path: Optional[Path]):
        if output_path is None:
            return

        symlink_path = Path(output_path).resolve() / "runs"

        if symlink_path.exists():
            if symlink_path.is_symlink() or symlink_path.is_dir():
                symlink_path.unlink()

        symlink_path.symlink_to(
            self.paths.runs,
            target_is_directory=True
        )