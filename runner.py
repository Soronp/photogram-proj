#!/usr/bin/env python3
"""
MARK-2 Pipeline Runner
----------------------

Deterministic.
Resume-capable.
Checkpoint-safe.
Compatible with new RunManager + RunContext.
"""

import argparse
import sys
import time
import traceback
import inspect
import json
from pathlib import Path
from typing import Callable, List, Tuple, Optional

from utils.paths import ProjectPaths
from utils.logger import get_run_logger
from runs.run_manager import RunManager

from init import run as run_init
from input import run as run_input
from image_analyzer import run as run_image_analyzer
from config_manager import create_runtime_config, validate_config
from filter import run as run_filter
from pre_proc import run as run_preprocess
from db_builder import run as run_db_builder
from matcher import run as run_matcher
from sparse_reconstruction import run as run_sparse
from sparse_eval import run as run_sparse_eval
from openmvs_export import run as run_openmvs_export
from dense_reconstruction import run as run_dense
from dense_eval import run as run_dense_eval
from dense_cleanup import run as run_dense_cleanup
from gen_mesh import run as run_mesh
from utils.mesh_cleanup import run as run_mesh_cleanup
from mesh_eval import run as run_mesh_eval
from eval_agg import run as run_eval_agg
from vis import run as run_vis


# ---------------------------------------------------------
# Pipeline Definition
# ---------------------------------------------------------

PipelineStage = Tuple[str, Callable | None]

PIPELINE: List[PipelineStage] = [
    ("init", run_init),
    ("input", run_input),
    ("image_analysis", run_image_analyzer),
    ("config", None),
    ("filter", run_filter),
    ("preprocess", run_preprocess),
    ("database", run_db_builder),
    ("matcher", run_matcher),
    ("sparse", run_sparse),
    ("sparse_eval", run_sparse_eval),
    ("openmvs_export", run_openmvs_export),
    ("dense", run_dense),
    ("dense_eval", run_dense_eval),
    ("dense_cleanup", run_dense_cleanup),
    ("mesh", run_mesh),
    ("mesh_cleanup", run_mesh_cleanup),
    ("mesh_eval", run_mesh_eval),
    ("aggregate_eval", run_eval_agg),
    ("visualization", run_vis),
]

PIPELINE_ORDER = [name for name, _ in PIPELINE]


# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------

def _invoke_stage(stage_fn: Callable, **context):
    sig = inspect.signature(stage_fn)
    kwargs = {k: context[k] for k in sig.parameters if k in context}
    return stage_fn(**kwargs)


def _write_config_snapshot(run_root: Path, config: dict, logger):
    snapshot = run_root / "config_snapshot.json"
    snapshot.write_text(json.dumps(config, indent=2))
    logger.info(f"[config] Snapshot written")


def _prompt_run_selection(run_manager: RunManager) -> Optional[str]:
    runs = run_manager.list_runs()

    if not runs:
        return None

    print("\nExisting runs:")
    for i, run_id in enumerate(runs, 1):
        print(f"{i}. {run_id}")

    print("0. Create new run")

    choice = input("\nSelect run: ").strip()

    if not choice or not choice.isdigit():
        return None

    idx = int(choice)

    if idx == 0:
        return None

    if 1 <= idx <= len(runs):
        return runs[idx - 1]

    return None


def _prompt_restart_stage(completed: List[str]) -> Optional[str]:
    if not completed:
        return None

    print("\nCompleted stages:")
    for stage in completed:
        print(f"  - {stage}")

    print("\nRestart options:")
    print("0. Resume normally")

    for i, stage in enumerate(PIPELINE_ORDER, 1):
        print(f"{i}. Restart from '{stage}'")

    choice = input("\nSelect option: ").strip()

    if not choice or not choice.isdigit():
        return None

    idx = int(choice)

    if idx == 0:
        return None

    if 1 <= idx <= len(PIPELINE_ORDER):
        return PIPELINE_ORDER[idx - 1]

    return None


# ---------------------------------------------------------
# Core Pipeline
# ---------------------------------------------------------

def run_pipeline(
    project_root: Path,
    input_path: Path,
    force: bool,
    output_symlink: Optional[Path],
):
    project_root = project_root.resolve()
    input_path = input_path.resolve()

    paths = ProjectPaths(project_root)
    paths.ensure_all()

    run_manager = RunManager(paths)

    # ---------------------------------------------
    # Run Selection
    # ---------------------------------------------

    selected_run = None if force else _prompt_run_selection(run_manager)

    if selected_run:
        run_ctx = run_manager.resume_run(selected_run, output_symlink)
    else:
        run_ctx = run_manager.start_new_run(
            project_root,
            input_path,
            output_symlink
        )

    logger = get_run_logger(run_ctx.run_id, run_ctx.logs)

    logger.info("=== MARK-2 PIPELINE STARTED ===")
    logger.info(f"Run ID     : {run_ctx.run_id}")
    logger.info(f"Input path : {input_path}")
    logger.info(f"Force      : {force}")

    # ---------------------------------------------
    # Resume Logic
    # ---------------------------------------------

    completed = run_ctx.completed_stages()
    restart_stage = None

    if completed and not force:
        restart_stage = _prompt_restart_stage(completed)

        if restart_stage:
            logger.info(f"[resume] Restarting from: {restart_stage}")
            run_ctx.clear_from(restart_stage, PIPELINE_ORDER)

    start_time = time.time()

    try:
        skip_until_restart = restart_stage is not None

        for stage_name, stage_fn in PIPELINE:

            if skip_until_restart:
                if stage_name != restart_stage:
                    continue
                skip_until_restart = False

            logger.info(f"[{stage_name}] START")

            if run_ctx.stage_done(stage_name) and not force:
                logger.info(f"[{stage_name}] SKIPPED (checkpoint)")
                continue

            try:
                if stage_name == "config":
                    config = create_runtime_config(
                        run_ctx.root,
                        project_root,
                        logger
                    )
                    validate_config(config, logger)
                    _write_config_snapshot(run_ctx.root, config, logger)
                else:
                    _invoke_stage(
                        stage_fn,
                        run_root=run_ctx.root,
                        project_root=project_root,
                        input_path=input_path,
                        force=force,
                        logger=logger,
                    )

                run_ctx.mark_stage(stage_name, "done")

            except Exception:
                run_ctx.mark_stage(stage_name, "failed")
                logger.error(f"[{stage_name}] FAILED")
                logger.error(traceback.format_exc())
                raise

            logger.info(f"[{stage_name}] DONE")

        elapsed = time.time() - start_time
        logger.info(f"PIPELINE SUCCESS ({elapsed:.2f}s)")
        run_manager.finish_run(success=True)

    except Exception:
        run_manager.finish_run(success=False)
        raise


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MARK-2 Pipeline Runner")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--project_root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    input_path = args.input or Path(input("Enter input path: ").strip())
    project_root = args.project_root or Path(input("Enter project root path: ").strip())
    output_symlink = args.output

    if not input_path.exists():
        print(f"Invalid input path: {input_path}")
        sys.exit(1)

    project_root.mkdir(parents=True, exist_ok=True)

    try:
        run_pipeline(project_root, input_path, args.force, output_symlink)
    except KeyboardInterrupt:
        print("\nPipeline interrupted")
        sys.exit(130)
    except Exception as exc:
        print(f"\nPipeline failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()