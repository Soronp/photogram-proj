#!/usr/bin/env python3
"""
runner.py

MARK-2 Pipeline Runner
---------------------
- Deterministic execution order
- Resume-safe
- UI-ready
- Accurate stage timing
- Matches actual function names and signatures
"""

import argparse
import sys
import time
import traceback
from pathlib import Path
from typing import Callable, List, Tuple

from utils.logger import get_logger
from utils.paths import ProjectPaths


# --------------------------------------------------
# Import pipeline stages (STRICT CONTRACT)
# --------------------------------------------------

from init import run_init
from input import run as run_input
from image_analyzer import run as run_image_analyzer
from config_manager import create_runtime_config
from filter import run as run_image_filter
from pre_proc import run as run_preprocess

from db_builder import run 
from matcher import run as run_matcher

from sparse_reconstruction import run_sparse
from sparse_eval import run_sparse_evaluation

from openmvs_export import run_openmvs_export
from dense_reconstruction import run_dense
from dense_eval import run_dense_evaluation

from gen_mesh import run_mesh_generation
from post_proc import run_mesh_postprocess
from gen_tex import run_texture_mapping

from mesh_eval import run_mesh_evaluation
from eval_agg import run_evaluation_aggregation
from vis import run_visualization


# --------------------------------------------------
# Canonical pipeline definition
# --------------------------------------------------

PipelineStage = Tuple[str, Callable[[Path, bool], None]]

PIPELINE: List[PipelineStage] = [
    ("init",           run_init),
    ("input",          None),  # special case
    ("image_analysis", run_image_analyzer),
    ("config",         lambda p, f: create_runtime_config(p)),
    ("filter",         run_image_filter),
    ("preprocess",     run_preprocess),

    ("database",       run),
    ("matcher",        run_matcher),

    ("sparse",         run_sparse),
    ("sparse_eval",    run_sparse_evaluation),

    ("openmvs_export", run_openmvs_export),

    ("dense",          run_dense),
    ("dense_eval",     run_dense_evaluation),

    ("mesh",           run_mesh_generation),
    ("mesh_post",      run_mesh_postprocess),
    ("texture",        run_texture_mapping),

    ("mesh_eval",      run_mesh_evaluation),
    ("aggregate_eval", run_evaluation_aggregation),
    ("visualization",  run_visualization),
]


# --------------------------------------------------
# Runner core
# --------------------------------------------------

def run_pipeline(input_path: Path, output_path: Path, force: bool):
    input_path = input_path.resolve()
    output_path = output_path.resolve()

    paths = ProjectPaths(output_path)
    paths.ensure_all()

    logger = get_logger("pipeline_runner", output_path)

    logger.info("MARK-2 pipeline started")
    logger.info(f"Input  : {input_path}")
    logger.info(f"Output : {output_path}")
    logger.info(f"Force  : {force}")

    total_stages = len(PIPELINE)
    pipeline_start = time.time()

    for idx, (stage_name, stage_fn) in enumerate(PIPELINE, start=1):
        stage_start = time.time()
        print(f"[STAGE_START] {idx}/{total_stages} {stage_name}", flush=True)

        try:
            if stage_name == "input":
                run_input(output_path, input_path, force)
            else:
                stage_fn(output_path, force)

        except Exception as exc:
            elapsed = time.time() - stage_start
            print(f"[STAGE_FAIL] {stage_name} ({elapsed:.2f}s)", flush=True)
            logger.error(traceback.format_exc())
            raise RuntimeError(
                f"Pipeline failed at stage '{stage_name}'"
            ) from exc

        elapsed = time.time() - stage_start
        print(f"[STAGE_DONE] {idx}/{total_stages} {stage_name} ({elapsed:.2f}s)", flush=True)
        logger.info(f"Stage '{stage_name}' completed in {elapsed:.2f}s")

    total_elapsed = time.time() - pipeline_start
    print(f"[PIPELINE_COMPLETE] Total time: {total_elapsed:.2f}s", flush=True)
    logger.info("MARK-2 pipeline completed successfully")


# --------------------------------------------------
# CLI
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MARK-2 Pipeline Runner")
    parser.add_argument("--input", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input) if args.input else Path(input("Enter input path: ").strip())
    output_path = Path(args.output) if args.output else Path(input("Enter output path: ").strip())

    if not input_path.exists():
        print(f"Invalid input path: {input_path}")
        sys.exit(1)

    output_path.mkdir(parents=True, exist_ok=True)

    try:
        run_pipeline(input_path, output_path, args.force)
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
        sys.exit(130)
    except Exception as exc:
        print(f"\nPipeline failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
