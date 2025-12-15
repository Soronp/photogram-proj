#!/usr/bin/env python3
"""
runner.py

MARK-2 Pipeline Runner with monitor support
Usage: python runner.py --input INPUT_PATH --output OUTPUT_PATH [--force]
"""

import argparse
import time
import traceback
from pathlib import Path
import yaml
import sys
import shutil

# -------------------------------
# Import pipeline stages (CALLABLES ONLY)
# -------------------------------

from init import run_init
from input import run as run_input
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
# Pipeline definition
# -------------------------------

PIPELINE = [
    ("init", run_init),
    ("input", run_input),
    ("image_filter", run_image_filter),
    ("preprocessing", run_preprocessing),
    ("database", run_db),
    ("matcher", run_matcher),
    ("sparse", run_sparse_reconstruction),
    ("sparse_eval", run_sparse_evaluation),
    ("dense", run_dense_reconstruction),
    ("dense_eval", lambda r, _: run_dense_evaluation(r)),
    ("mesh", run_mesh_generation),
    ("mesh_post", run_mesh_postprocess),
    ("texture", run_texture_mapping),
    ("mesh_eval", lambda r, _: run_mesh_evaluation(r)),
    ("aggregate", lambda r, _: run_evaluation_aggregation(r)),
    ("visualization", lambda r, _: run_visualization(r)),
]

# -------------------------------
# Simple config generator - CREATED FIRST
# -------------------------------

def create_config_first(output_path: Path):
    """Create config.yaml FIRST before anything else"""
    config_path = output_path / "config.yaml"
    
    if config_path.exists():
        return config_path
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    simple_config = {
        "project_name": output_path.name,
        "input_format": "jpg",
        "output_format": "ply",
        "dense_max_image_size": 2400,
        "patchmatch_iterations": 3,
        "patchmatch_samples": 25,
        "texture_size": 2048,
    }
    
    with open(config_path, "w") as f:
        yaml.dump(simple_config, f)
    
    return config_path

# -------------------------------
# Setup project structure
# -------------------------------

def setup_project_structure(input_path: Path, output_path: Path):
    """Setup required structure for the pipeline - AFTER config is created"""
    # Create images directory
    images_dir = output_path / "images"
    if not images_dir.exists():
        try:
            shutil.copytree(input_path, images_dir)
        except Exception as e:
            images_dir.mkdir(exist_ok=True)
    
    return output_path

# -------------------------------
# MONITOR-FRIENDLY PRINT FUNCTIONS
# -------------------------------

def monitor_print(message):
    """Print message for monitor with immediate flush"""
    print(message, flush=True)

def monitor_start_stage(idx, name, total_stages):
    """Print stage start marker for monitor"""
    monitor_print(f"[STAGE_START] {idx}/{total_stages} {name}")

def monitor_end_stage(idx, name, elapsed, total_stages):
    """Print stage end marker for monitor"""
    monitor_print(f"[STAGE_DONE] {idx}/{total_stages} {name} {elapsed:.2f}s")

def monitor_fail_stage(name, error):
    """Print stage failure marker for monitor"""
    monitor_print(f"[STAGE_FAIL] {name}: {error}")

def monitor_pipeline_complete(output_path):
    """Print pipeline completion marker for monitor"""
    monitor_print(f"[PIPELINE_COMPLETE] Results in: {output_path}")

# -------------------------------
# Run pipeline
# -------------------------------

def run_pipeline_from_args(input_path: Path, output_path: Path, force: bool = False):
    """Run pipeline with input/output paths as arguments"""
    monitor_print(f"[PIPELINE_START] {input_path} -> {output_path}")
    monitor_print(f"[TOTAL_STAGES] {len(PIPELINE)}")
    
    # STEP 1: Create config.yaml FIRST
    try:
        create_config_first(output_path)
    except Exception as e:
        monitor_print(f"[STAGE_FAIL] config_creation: {str(e)}")
        sys.exit(1)
    
    # STEP 2: Setup project structure
    try:
        project_root = setup_project_structure(input_path, output_path)
    except Exception as e:
        monitor_print(f"[STAGE_FAIL] project_setup: {str(e)}")
        sys.exit(1)
    
    # STEP 3: Import project modules
    try:
        from utils.logger import get_logger
        from utils.paths import ProjectPaths
    except ImportError as e:
        monitor_print(f"[STAGE_FAIL] module_import: {str(e)}")
        sys.exit(1)
    
    # STEP 4: Create all project paths
    try:
        paths = ProjectPaths(project_root)
        paths.ensure_all()
    except Exception as e:
        monitor_print(f"[STAGE_FAIL] path_creation: {str(e)}")
        sys.exit(1)
    
    # STEP 5: Setup logger (logs go to file, not to monitor)
    logger = get_logger("pipeline_runner", project_root)
    logger.info("=== MARK-2 PIPELINE START ===")
    logger.info(f"Input folder : {input_path}")
    logger.info(f"Output folder: {output_path}")
    logger.info(f"Project root : {project_root}")
    logger.info(f"Force mode   : {force}")
    
    # STEP 6: Run pipeline stages
    for idx, (name, func) in enumerate(PIPELINE, 1):
        monitor_start_stage(idx, name, len(PIPELINE))
        logger.info(f"[STAGE START] {name}")
        start = time.time()
        
        try:
            func(project_root, force)
        except Exception as e:
            monitor_fail_stage(name, str(e))
            logger.error(f"[STAGE FAIL] {name}")
            logger.error(traceback.format_exc())
            raise RuntimeError(f"Pipeline aborted at stage: {name}") from e
        
        elapsed = time.time() - start
        monitor_end_stage(idx, name, elapsed, len(PIPELINE))
        logger.info(f"[STAGE DONE] {name} ({elapsed:.2f}s)")
    
    # STEP 7: Completion
    logger.info("=== MARK-2 PIPELINE COMPLETE ===")
    monitor_pipeline_complete(output_path)

def clean_path_string(path_str: str) -> Path:
    """Clean up path string - remove quotes and normalize"""
    path_str = path_str.strip('"\'')
    if path_str.endswith('"'):
        path_str = path_str[:-1]
    path_str = path_str.rstrip('\\/')
    return Path(path_str).resolve()

def main():
    """Command line entry point"""
    parser = argparse.ArgumentParser(
        description='MARK-2 Photogrammetry Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python runner.py --input "D:\\input\\images" --output "D:\\output"
  python runner.py --input "D:/input/images/" --output "D:/output/"
        """
    )
    parser.add_argument('--input', required=True, help='Input folder with images')
    parser.add_argument('--output', required=True, help='Output folder for results')
    parser.add_argument('--force', action='store_true', help='Force re-run all stages')
    
    args = parser.parse_args()
    
    # Clean and validate paths
    try:
        input_path = clean_path_string(args.input)
        output_path = clean_path_string(args.output)
    except Exception as e:
        print(f"Error parsing paths: {e}", flush=True)
        sys.exit(1)
    
    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}", flush=True)
        sys.exit(1)
    
    if not input_path.is_dir():
        print(f"Error: Input path is not a directory: {input_path}", flush=True)
        sys.exit(1)
    
    try:
        run_pipeline_from_args(input_path, output_path, args.force)
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user", flush=True)
        sys.exit(130)
    except Exception as e:
        print(f"\nPipeline failed: {e}", flush=True)
        sys.exit(1)

if __name__ == "__main__":
    main()