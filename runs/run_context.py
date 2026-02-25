#!/usr/bin/env python3
"""
RunContext (Authoritative + Deterministic)

Owns:
- Run identity
- Manifest metadata
- Checkpoints
- Logs directory

Immutable contract:
Pipeline stages MUST NOT modify this object directly.
Only runner controls checkpoints.
"""

from pathlib import Path
from datetime import datetime
import json
import uuid
import time
from typing import List, Dict, Optional


class RunContext:

    # --------------------------------------------------
    # Construction
    # --------------------------------------------------

    def __init__(self, runs_root: Path, run_id: Optional[str] = None):
        self.runs_root = runs_root.resolve()
        self.runs_root.mkdir(parents=True, exist_ok=True)

        if run_id:
            # Resume existing
            self.run_id = run_id
            self.root = self.runs_root / run_id
            if not self.root.exists():
                raise RuntimeError(f"Run not found: {self.root}")
            self._new_run = False
        else:
            # Create new
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            uid = uuid.uuid4().hex[:6]
            self.run_id = f"run_{ts}_{uid}"
            self.root = self.runs_root / self.run_id
            self._new_run = True

        self.logs = self.root / "logs"
        self.manifest = self.root / "run.json"
        self.checkpoints = self.root / "checkpoints.json"

        self._start_time: Optional[float] = None

    # --------------------------------------------------
    # Initialization (new run only)
    # --------------------------------------------------

    def initialize(self, project_root: Path, input_path: Path):
        if not self._new_run:
            raise RuntimeError("Cannot initialize an existing run")

        if self.root.exists():
            raise RuntimeError(f"Run already exists: {self.root}")

        self.root.mkdir(parents=True)
        self.logs.mkdir()

        self._start_time = time.time()

        manifest_data = {
            "run_id": self.run_id,
            "pipeline": "MARK-2",
            "status": "running",
            "started_at": datetime.utcnow().isoformat() + "Z",
            "project_root": str(project_root.resolve()),
            "input_path": str(input_path.resolve()),
        }

        self.manifest.write_text(json.dumps(manifest_data, indent=2))
        self.checkpoints.write_text(json.dumps({}, indent=2))

    # --------------------------------------------------
    # Resume Validation
    # --------------------------------------------------

    def validate(self):
        if not self.manifest.exists():
            raise RuntimeError("Run manifest missing")

        if not self.checkpoints.exists():
            raise RuntimeError("Checkpoint file missing")

    # --------------------------------------------------
    # Checkpoints (Runner Only)
    # --------------------------------------------------

    def _load_checkpoints(self) -> Dict[str, str]:
        if not self.checkpoints.exists():
            return {}
        return json.loads(self.checkpoints.read_text())

    def _write_checkpoints(self, data: Dict[str, str]):
        self.checkpoints.write_text(json.dumps(data, indent=2))

    def mark_stage(self, stage: str, status: str):
        if status not in {"done", "failed"}:
            raise ValueError("Invalid stage status")

        data = self._load_checkpoints()
        data[stage] = status
        self._write_checkpoints(data)

    def stage_done(self, stage: str) -> bool:
        return self._load_checkpoints().get(stage) == "done"

    def completed_stages(self) -> List[str]:
        return [
            k for k, v in self._load_checkpoints().items()
            if v == "done"
        ]

    def clear_from(self, restart_stage: str, pipeline_order: List[str]):
        data = self._load_checkpoints()

        if restart_stage not in pipeline_order:
            raise ValueError(f"Unknown stage: {restart_stage}")

        clear = False
        for stage in pipeline_order:
            if stage == restart_stage:
                clear = True
            if clear and stage in data:
                del data[stage]

        self._write_checkpoints(data)

    # --------------------------------------------------
    # Finalization
    # --------------------------------------------------

    def finalize(self, success: bool):
        if not self.manifest.exists():
            raise RuntimeError("Run manifest missing")

        manifest = json.loads(self.manifest.read_text())

        manifest["status"] = "success" if success else "failed"
        manifest["finished_at"] = datetime.utcnow().isoformat() + "Z"

        if self._start_time is not None:
            manifest["duration_sec"] = round(
                time.time() - self._start_time, 3
            )

        self.manifest.write_text(json.dumps(manifest, indent=2))