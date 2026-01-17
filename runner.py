#!/usr/bin/env python3
"""
runner.py

MARK-2 Pipeline Runner (Authoritative)
-------------------------------------
- Runs are created under PROJECT_ROOT/runs/<run_id>
- Exactly one logger per run
- Deterministic stage execution
- Resume-safe via checkpoints
"""

import argparse
import sys
import time
import traceback
from pathlib import Path
from typing import Callable, List, Tuple

from utils.paths import ProjectPaths
from utils.logger import get_run_logger
from runs.run_manager import RunManager

# --------------------------------------------------
# Import pipeline stages
# --------------------------------------------------
from init import run as run_init
from input import run as run_input
from image_analyzer import run as run_image_analyzer
from config_manager import create_runtime_config, validate_config
from filter import run as run_image_filter
from pre_proc import run as run_preprocess
from db_builder import run as run_db_builder
from matcher import run as run_matcher
from sparse_reconstruction import run as run_sparse
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
PipelineStage = Tuple[str, Callable]

PIPELINE: List[PipelineStage] = [
    ("init", run_init),
    ("input", None),
    ("image_analysis", run_image_analyzer),
    ("config", None),
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
def run_pipeline(
    project_root: Path,
    input_path: Path,
    force: bool,
    output_symlink: Path | None = None,
):
    project_root = project_root.resolve()
    input_path = input_path.resolve()

    paths = ProjectPaths(project_root)
    paths.ensure_all()

    run_manager = RunManager(paths)
    run_ctx = run_manager.start_run(project_root, input_path)

    logger = get_run_logger(run_ctx.run_id, run_ctx.logs)

    logger.info("=== MARK-2 PIPELINE STARTED ===")
    logger.info(f"Run ID       : {run_ctx.run_id}")
    logger.info(f"Project root : {project_root}")
    logger.info(f"Input path   : {input_path}")
    logger.info(f"Force        : {force}")

    pipeline_start = time.time()

    try:
        for stage_name, stage_fn in PIPELINE:
            logger.info(f"[{stage_name}] START")

            if run_ctx.stage_done(stage_name) and not force:
                logger.info(f"[{stage_name}] SKIPPED (checkpoint)")
                continue

            try:
                if stage_name == "input":
                    run_input(project_root, input_path, force, logger)

                elif stage_name == "config":
                    config = create_runtime_config(run_ctx.root, project_root, logger)
                    if not validate_config(config, logger):
                        raise RuntimeError("Configuration validation failed")

                else:
                    stage_fn(run_ctx.root, project_root, force, logger)

                run_ctx.mark_stage(stage_name, "done")

            except Exception:
                logger.error(f"[{stage_name}] FAILED")
                logger.error(traceback.format_exc())
                raise

            logger.info(f"[{stage_name}] DONE")

        logger.info(f"PIPELINE SUCCESS ({time.time() - pipeline_start:.2f}s)")
        run_manager.finish_run(success=True)

        if output_symlink:
            try:
                if output_symlink.exists() or output_symlink.is_symlink():
                    output_symlink.unlink()
                output_symlink.symlink_to(run_ctx.root, target_is_directory=True)
                logger.info(f"Symlink created: {output_symlink} -> {run_ctx.root}")
            except Exception as e:
                logger.warning(f"Symlink creation failed: {e}")

    except Exception:
        run_manager.finish_run(success=False)
        raise


# --------------------------------------------------
# CLI
# --------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="MARK-2 Pipeline Runner")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--project_root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    input_path = args.input or Path(input("Enter input path: ").strip())
    project_root = args.project_root or Path(input("Enter project root path: ").strip())
    output_symlink = args.output.resolve() if args.output else None

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
