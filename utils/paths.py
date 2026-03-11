#!/usr/bin/env python3
"""
paths.py

Filesystem layout manager for the MARK-2 pipeline.

Design goals
------------
• deterministic directory structure
• isolated runs
• reproducible outputs
"""

from pathlib import Path


# --------------------------------------------------
# Workspace Paths
# --------------------------------------------------

class WorkspacePaths:
    """
    Root workspace containing all runs.
    """

    def __init__(self, root: Path):

        self.root = Path(root).resolve()

        self.runs = self.root / "runs"

    def ensure(self):

        self.root.mkdir(parents=True, exist_ok=True)
        self.runs.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Run Paths
# --------------------------------------------------

class RunPaths:
    """
    Directory structure for a single pipeline run.
    """

    def __init__(self, workspace: WorkspacePaths, run_id: str):

        self.run_id = run_id

        self.root = workspace.runs / run_id

        # -----------------------------
        # INPUT
        # -----------------------------

        self.input = self.root / "input"
        self.images = self.input / "images"
        self.videos = self.input / "videos"

        # -----------------------------
        # IMAGE PROCESSING
        # -----------------------------

        self.images_preprocessed = self.root / "images_preprocessed"
        self.images_filtered = self.root / "images_filtered"

        # -----------------------------
        # COLMAP
        # -----------------------------

        self.database = self.root / "database"
        self.database_path = self.database / "database.db"

        self.sparse = self.root / "sparse"

        # -----------------------------
        # OPENMVS / DENSE
        # -----------------------------

        self.openmvs = self.root / "openmvs"
        self.dense = self.root / "dense"

        # -----------------------------
        # OUTPUT ASSETS
        # -----------------------------

        self.mesh = self.root / "mesh"
        self.textures = self.root / "textures"

        # -----------------------------
        # ANALYSIS
        # -----------------------------

        self.evaluation = self.root / "evaluation"
        self.visualization = self.root / "visualization"

        # -----------------------------
        # LOGGING
        # -----------------------------

        self.logs = self.root / "logs"

    # --------------------------------------------------

    def ensure(self):
        """
        Create directory structure for the run.
        """

        dirs = [

            self.root,

            self.input,
            self.images,
            self.videos,

            self.images_preprocessed,
            self.images_filtered,

            self.database,
            self.sparse,

            self.openmvs,
            self.dense,

            self.mesh,
            self.textures,

            self.evaluation,
            self.visualization,

            self.logs,
        ]

        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)