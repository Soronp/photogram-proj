#!/usr/bin/env python3
"""
MARK-2 Photogrammetry Pipeline Runner
Deterministic execution engine
"""

import sys
import time
import traceback
import argparse
import importlib
from pathlib import Path

# --------------------------------------------------
# Make project root importable
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# --------------------------------------------------

from utils.paths import WorkspacePaths, RunPaths
from utils.logger import create_run_logger
from core.tool_runner import ToolRunner
from config.config_manager import create_runtime_config, validate_config


# --------------------------------------------------
# MARK-2 Pipeline Stage Order
# --------------------------------------------------

PIPELINE_STAGES = [

    # DATASET PREPARATION
    "stages.dataset_manifest",
    "stages.ingestion",
    "stages.pre_proc",
    "stages.filter",

    # SPARSE RECONSTRUCTION
    "stages.sparse.db_builder",
    "stages.sparse.matcher",
    "stages.sparse.sparse_reconstruction",
    "stages.sparse.sparse_eval",

    # DENSE RECONSTRUCTION
    "stages.dense.openmvs_export",
    "stages.dense.dense_reconstruction",
    "stages.dense.dense_cleanup",
    "stages.dense.dense_eval",

    # MESH
    "stages.mesh.gen_mesh",

    # MESH EVALUATION
    "stages.mesh.mesh_eval",

    # TEXTURE
    "stages.mesh.gen_tex",

    # GLOBAL EVALUATION
    "evaluation.eval_agg"
]


# --------------------------------------------------
# Stage Loader
# --------------------------------------------------

def load_stage(stage_path: str, dev_reload: bool = False):

    try:

        module = importlib.import_module(stage_path)

        if dev_reload:
            module = importlib.reload(module)

    except Exception as e:

        raise RuntimeError(
            f"Failed to import stage: {stage_path}"
        ) from e

    if not hasattr(module, "run"):

        raise RuntimeError(
            f"{stage_path} missing run()"
        )

    return module.run


# --------------------------------------------------
# CLI Prompts
# --------------------------------------------------

def prompt_dataset():

    while True:

        p = input("\nDataset folder: ").strip()
        path = Path(p).expanduser().resolve()

        if path.exists():
            return path

        print("❌ Path does not exist")


def prompt_workspace():

    while True:

        p = input("\nWorkspace folder: ").strip()
        path = Path(p).expanduser().resolve()

        try:
            path.mkdir(parents=True, exist_ok=True)
            return path
        except Exception:
            print("❌ Cannot create workspace")


# --------------------------------------------------
# Pipeline Runner
# --------------------------------------------------

class PipelineRunner:

    def __init__(self, dataset_path: Path, workspace_root: Path):

        if not dataset_path.exists():
            raise RuntimeError("Dataset path does not exist")

        self.dataset_path = dataset_path

        # Workspace
        self.workspace = WorkspacePaths(workspace_root)
        self.workspace.ensure()

        run_id = time.strftime("run_%Y%m%d_%H%M%S")

        self.paths = RunPaths(self.workspace, run_id)
        self.paths.ensure()

        # Logger
        self.logger = create_run_logger(
            run_id,
            self.paths.logs
        )

        self.logger.info("PIPELINE START")
        self.logger.info(f"run_id: {run_id}")
        self.logger.info(f"dataset: {dataset_path}")

        # Config
        self.config = create_runtime_config(
            self.paths.root,
            dataset_path,
            self.logger
        )

        validate_config(self.config, self.logger)

        # Tool runner
        self.tools = ToolRunner(
            self.config,
            self.logger
        )

        self.dev_reload = self.config.get("dev_reload", False)

    # --------------------------------------------------

    def run_stage(self, stage_path: str, index: int, total: int):

        stage_name = stage_path.split(".")[-1]

        self.logger.info(
            f"[stage {index}/{total}] {stage_name} START"
        )

        start = time.time()

        try:

            stage_fn = load_stage(
                stage_path,
                dev_reload=self.dev_reload
            )

            stage_fn(
                paths=self.paths,
                tools=self.tools,
                config=self.config,
                logger=self.logger
            )

        except Exception:

            self.logger.error(
                f"[stage {index}/{total}] {stage_name} FAILED"
            )

            self.logger.error(traceback.format_exc())

            raise RuntimeError(
                f"Pipeline stopped at stage: {stage_name}"
            )

        elapsed = time.time() - start

        self.logger.info(
            f"[stage {index}/{total}] {stage_name} DONE ({elapsed:.2f}s)"
        )

    # --------------------------------------------------

    def execute(self):

        pipeline_start = time.time()

        total = len(PIPELINE_STAGES)

        try:

            for i, stage in enumerate(PIPELINE_STAGES, start=1):

                self.run_stage(stage, i, total)

        except RuntimeError:

            elapsed = time.time() - pipeline_start

            self.logger.error(
                f"PIPELINE FAILED after {elapsed:.2f}s"
            )

            raise

        elapsed = time.time() - pipeline_start

        self.logger.info(
            f"PIPELINE COMPLETE ({elapsed:.2f}s)"
        )


# --------------------------------------------------
# Entry Point
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str)
    parser.add_argument("--workspace", type=str)

    args = parser.parse_args()

    dataset = (
        Path(args.dataset).resolve()
        if args.dataset
        else prompt_dataset()
    )

    workspace = (
        Path(args.workspace).resolve()
        if args.workspace
        else prompt_workspace()
    )

    runner = PipelineRunner(dataset, workspace)

    runner.execute()


if __name__ == "__main__":
    main()