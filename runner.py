#!/usr/bin/env python3
"""
runner.py

MARK-2 Pipeline Runner
---------------------
- Deterministic execution order
- Resume-safe
- UI-ready
- Accurate stage timing
"""

import argparse
import sys
import time
import traceback
from pathlib import Path

from utils.logger import get_logger
from utils.paths import ProjectPaths

# -------------------------------
# Import pipeline stages
# -------------------------------
from init import run_init
from input import run as run_input
from image_analyzer import run as run_image_analyzer
from config_manager import create_runtime_config
from filter import run as run_image_filter
from pre_proc import run as run_preprocessing

from db_builder import run as run_db
from matcher import run as run_matcher
from sparse_reconstruction import run_sparse_reconstruction
from sparse_eval import run_sparse_evaluation
from dense_reconstruction import run_dense_reconstruction
from dense_eval import run_dense_evaluation
from gen_mesh import run_mesh_generation
from post_proc import run_mesh_postprocess
from gen_tex import run_texture_mapping
from mesh_eval import run_mesh_evaluation
from eval_agg import run_evaluation_aggregation
from vis import run_visualization


# -------------------------------
# Canonical pipeline definition
# -------------------------------
PIPELINE = [
    ("init", lambda p, f: run_init(p, f)),
    ("input", lambda p, f: run_input(p, None, f)),
    ("image_analysis", lambda p, f: run_image_analyzer(p, f)),
    ("config", lambda p, f: create_runtime_config(p)),
    ("filter", lambda p, f: run_image_filter(p, f)),
    ("preprocess", lambda p, f: run_preprocessing(p, f)),

    ("database", lambda p, f: run_db(p, f)),
    ("matcher", lambda p, f: run_matcher(p, f)),
    ("sparse", lambda p, f: run_sparse_reconstruction(p, f)),
    ("sparse_eval", lambda p, f: run_sparse_evaluation(p, f)),
    ("dense", lambda p, f: run_dense_reconstruction(p, f)),
    ("dense_eval", lambda p, f: run_dense_evaluation(p, f)),
    ("mesh", lambda p, f: run_mesh_generation(p, f)),
    ("mesh_post", lambda p, f: run_mesh_postprocess(p, f)),
    ("texture", lambda p, f: run_texture_mapping(p, f)),
    ("mesh_eval", lambda p, f: run_mesh_evaluation(p, f)),
    ("aggregate_eval", lambda p, f: run_evaluation_aggregation(p, f)),
    ("visualization", lambda p, f: run_visualization(p, f)),
]


# -------------------------------
# Runner
# -------------------------------
def run_pipeline(input_path: Path, output_path: Path, force: bool):
    input_path = input_path.resolve()
    output_path = output_path.resolve()

    paths = ProjectPaths(output_path)
    paths.ensure_all()

    logger = get_logger("pipeline_runner", output_path)
    logger.info(f"Input : {input_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Force : {force}")

    total_stages = len(PIPELINE)
    pipeline_start = time.time()

    for idx, (name, stage_fn) in enumerate(PIPELINE, start=1):
        stage_start = time.time()
        print(f"[STAGE_START] {idx}/{total_stages} {name}", flush=True)

        try:
            # Special case: input stage needs input_path
            if name == "input":
                run_input(output_path, input_path, force)
            else:
                stage_fn(output_path, force)

        except Exception as e:
            print(f"[STAGE_FAIL] {name}: {e}", flush=True)
            logger.error(traceback.format_exc())
            raise RuntimeError(f"Pipeline failed at stage '{name}'") from e

        elapsed = time.time() - stage_start
        print(f"[STAGE_DONE] {idx}/{total_stages} {name} ({elapsed:.2f}s)", flush=True)

    total_elapsed = time.time() - pipeline_start
    print(f"[PIPELINE_COMPLETE] Total time: {total_elapsed:.2f}s", flush=True)
    logger.info("MARK-2 pipeline completed successfully")


# -------------------------------
# CLI
# -------------------------------
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
    except Exception as e:
        print(f"\nPipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
