from pathlib import Path
from datetime import datetime
import json
import uuid
import time


class RunContext:
    """
    MARK-2 Run Context (Authoritative)

    One instance per pipeline execution.
    Owns:
      - run identity
      - run-scoped logs
      - checkpoints
      - immutable run metadata

    NO pipeline stage may create or modify this.
    """

    def __init__(self, runs_root: Path):
        runs_root = runs_root.resolve()

        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        uid = uuid.uuid4().hex[:6]

        self.run_id = f"run_{ts}_{uid}"

        # -----------------------------
        # Authoritative run locations
        # -----------------------------
        self.root: Path = runs_root / self.run_id
        self.logs: Path = self.root / "logs"

        self.checkpoints: Path = self.root / "checkpoints.json"
        self.manifest: Path = self.root / "run.json"

        self._start_time: float | None = None

    # --------------------------------------------------
    # Lifecycle
    # --------------------------------------------------

    def initialize(self, project_root: Path, input_path: Path):
        """
        Create run directories and immutable metadata.
        Must be called exactly once.
        """
        if self.root.exists():
            raise RuntimeError(f"Run already exists: {self.root}")

        self.root.mkdir(parents=True)
        self.logs.mkdir()

        self._start_time = time.time()

        with open(self.manifest, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "run_id": self.run_id,
                    "status": "running",
                    "started_at": datetime.utcnow().isoformat() + "Z",
                    "project_root": str(project_root.resolve()),
                    "input_path": str(input_path.resolve()),
                    "pipeline": "MARK-2",
                },
                f,
                indent=2,
            )

        with open(self.checkpoints, "w", encoding="utf-8") as f:
            json.dump({}, f, indent=2)

    # --------------------------------------------------
    # Checkpoints (runner-only)
    # --------------------------------------------------

    def mark_stage(self, stage: str, status: str):
        """
        status ∈ {"done", "failed"}
        """
        if not self.checkpoints.exists():
            raise RuntimeError("Checkpoints file missing")

        with open(self.checkpoints, "r+", encoding="utf-8") as f:
            data = json.load(f)
            data[stage] = status
            f.seek(0)
            json.dump(data, f, indent=2)
            f.truncate()

    def stage_done(self, stage: str) -> bool:
        if not self.checkpoints.exists():
            return False

        with open(self.checkpoints, encoding="utf-8") as f:
            return json.load(f).get(stage) == "done"

    # --------------------------------------------------
    # Finalization
    # --------------------------------------------------

    def finalize(self, success: bool):
        if not self.manifest.exists():
            raise RuntimeError("Run manifest missing")

        finished_at = datetime.utcnow().isoformat() + "Z"
        duration = None

        if self._start_time is not None:
            duration = round(time.time() - self._start_time, 3)

        with open(self.manifest, "r+", encoding="utf-8") as f:
            data = json.load(f)
            data["status"] = "success" if success else "failed"
            data["finished_at"] = finished_at
            data["duration_sec"] = duration
            f.seek(0)
            json.dump(data, f, indent=2)
            f.truncate()
