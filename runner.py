#!/usr/bin/env python3
"""
runner.py

MARK-2 Pipeline Runner (Project-Scoped Runs with Output Symlink)
----------------------------------------------------------------
- All runs are stored under PROJECT_ROOT/runs/
- Optional symlink in OUTPUT folder for quick access
- Deterministic execution order
- Resume-safe by outputs
- Single logger per run
"""

import argparse
import sys
import time
import traceback
from pathlib import Path
from typing import Callable, List, Tuple

from utils.logger import get_logger
from utils.paths import ProjectPaths
from runs.run_manager import RunManager

# --------------------------------------------------
# Import pipeline stages
# --------------------------------------------------
from init import run as run_init
from input import run as run_input
from image_analyzer import run as run_image_analyzer
from config_manager import create_runtime_config
from filter import run as run_image_filter
from pre_proc import run as run_preprocess
from db_builder import run as run_db_builder
from matcher import run as run_matcher
from sparse_reconstruction import run_sparse
from sparse_eval import run as run_sparse_eval
from openmvs_export import run as run_openmvs_export
from dense_reconstruction import run as run_dense
from dense_eval import run as run_dense_evaluation
from gen_mesh import run as run_mesh_generation
from utils.mesh_cleanup import run as run_mesh_cleanup
from mesh_eval import run as run_mesh_evaluation
from eval_agg import run as run_eval_agg
from vis import run as run_visualization

# --------------------------------------------------
# Pipeline definition
# --------------------------------------------------
PipelineStage = Tuple[str, Callable | None]

PIPELINE: List[PipelineStage] = [
    ("init", run_init),
    ("input", None),  # special case
    ("image_analysis", run_image_analyzer),
    ("config", lambda p, f, logger: create_runtime_config(p, logger)),
    ("filter", run_image_filter),
    ("preprocess", run_preprocess),
    ("database", run_db_builder),
    ("matcher", run_matcher),
    ("sparse", run_sparse),
    ("sparse_eval", run_sparse_eval),
    ("openmvs_export", run_openmvs_export),
    ("dense", run_dense),
    ("dense_eval", run_dense_evaluation),
    ("mesh", run_mesh_generation),
    ("mesh_cleanup", run_mesh_cleanup),
    ("mesh_eval", run_mesh_evaluation),
    ("aggregate_eval", run_eval_agg),
    ("visualization", run_visualization),
]

# --------------------------------------------------
# Runner core
# --------------------------------------------------
def run_pipeline(project_root: Path, input_path: Path, force: bool, output_symlink: Path | None = None):
    project_root = project_root.resolve()
    input_path = input_path.resolve()

    # Initialize canonical project paths
    paths = ProjectPaths(project_root)
    paths.ensure_all()

    # Create persistent run directory under PROJECT_ROOT/runs/
    run_manager = RunManager(paths)
    run_ctx = run_manager.start_run(project_root, input_path)
    run_log_dir = run_ctx.logs  # PROJECT_ROOT/runs/<run_id>/logs

    # Single logger for entire run
    logger = get_logger("run", log_root=run_log_dir)

    logger.info("=== MARK-2 PIPELINE STARTED ===")
    logger.info(f"Run ID       : {run_ctx.run_id}")
    logger.info(f"Project root : {project_root}")
    logger.info(f"Input path   : {input_path}")
    logger.info(f"Force        : {force}")

    total_stages = len(PIPELINE)
    pipeline_start = time.time()

    try:
        for idx, (stage_name, stage_fn) in enumerate(PIPELINE, start=1):
            stage_start = time.time()
            print(f"[STAGE_START] {idx}/{total_stages} {stage_name}", flush=True)
            logger.info(f"[{stage_name}] START")

            # Skip if checkpoint exists and force is not enabled
            if run_ctx.stage_done(stage_name) and not force:
                logger.info(f"[{stage_name}] SKIPPED (checkpoint)")
                print(f"[STAGE_SKIP] {idx}/{total_stages} {stage_name}", flush=True)
                continue

            try:
                # Special input stage handling
                if stage_name == "input":
                    run_input(project_root, input_path, force, logger)
                elif stage_fn is not None:
                    stage_fn(project_root, force, logger)

                run_ctx.mark_stage(stage_name, "done")

            except Exception:
                logger.error(f"[{stage_name}] FAILED")
                logger.error(traceback.format_exc())
                raise

            elapsed = time.time() - stage_start
            logger.info(f"[{stage_name}] DONE in {elapsed:.2f}s")
            print(f"[STAGE_DONE] {idx}/{total_stages} {stage_name} ({elapsed:.2f}s)", flush=True)

        total_elapsed = time.time() - pipeline_start
        logger.info(f"PIPELINE SUCCESS ({total_elapsed:.2f}s)")
        run_manager.finish_run(success=True)

        # --------------------------------------------------
        # Optional symlink creation in output folder
        # --------------------------------------------------
        if output_symlink:
            try:
                if output_symlink.exists() or output_symlink.is_symlink():
                    output_symlink.unlink()
                output_symlink.symlink_to(run_ctx.run_dir, target_is_directory=True)
                logger.info(f"Symlink created at {output_symlink} -> {run_ctx.run_dir}")
            except Exception as e:
                logger.warning(f"Could not create symlink at {output_symlink}: {e}")

    except Exception:
        run_manager.finish_run(success=False)
        raise

# --------------------------------------------------
# CLI
# --------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="MARK-2 Pipeline Runner")
    parser.add_argument("--input", type=Path, help="Input data directory")
    parser.add_argument("--project_root", type=Path, help="Project root directory (canonical)")
    parser.add_argument("--output", type=Path, help="Optional output folder for symlink")
    parser.add_argument("--force", action="store_true", help="Force re-run of stages")
    args = parser.parse_args()

    input_path = args.input or Path(input("Enter input path: ").strip())
    project_root = args.project_root or Path(input("Enter project root path: ").strip())
    output_symlink = args.output.resolve() if args.output else None

    if not input_path.exists():
        print(f"Invalid input path: {input_path}")
        sys.exit(1)

    project_root.mkdir(parents=True, exist_ok=True)
    if output_symlink:
        output_symlink.parent.mkdir(parents=True, exist_ok=True)

    try:
        run_pipeline(project_root, input_path, args.force, output_symlink)
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
        sys.exit(130)
    except Exception as exc:
        print(f"\nPipeline failed: {exc}")
        sys.exit(1)

if __name__ == "__main__":
    main()
