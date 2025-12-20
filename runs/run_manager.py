#!/usr/bin/env python3
"""
RunManager for MARK-2
---------------------
- Creates and manages project-scoped runs
- Primary run directory: PROJECT_ROOT/runs/<run_id>
- Optionally creates symlink in output folder
"""
from pathlib import Path
from runs.run_context import RunContext
from utils.paths import ProjectPaths

class RunManager:
    def __init__(self, project_paths: 'ProjectPaths'):
        self.paths = project_paths
        self.active_run: RunContext | None = None

    def start_run(self, project_root: Path, input_path: Path, output_path: Path | None = None) -> RunContext:
        """
        Initialize a new run.

        - Primary run created under PROJECT_ROOT/runs/<run_id>
        - Optional symlink at output_path/runs -> PROJECT_ROOT/runs

        Returns:
            RunContext
        """
        if self.active_run is not None:
            raise RuntimeError("A run is already active")

        # Primary run directory
        self.active_run = RunContext(self.paths.runs)
        self.active_run.initialize(project_root, input_path)

        # Optional symlink for quick access
        if output_path is not None:
            symlink_path = Path(output_path).resolve() / "runs"
            if symlink_path.exists():
                if symlink_path.is_symlink() or symlink_path.is_dir():
                    symlink_path.unlink()
            symlink_path.symlink_to(self.paths.runs, target_is_directory=True)

        return self.active_run

    def finish_run(self, success: bool):
        """
        Finalize the active run.
        """
        if self.active_run:
            self.active_run.finalize(success)
            self.active_run = None
