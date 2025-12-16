#!/usr/bin/env python3
"""
input.py

Stage: Copy user-provided input images/videos into the project raw/ folder.
Only creates project structure in the project root (output folder).
"""

from pathlib import Path
import shutil
from utils.logger import get_logger
from utils.paths import ProjectPaths

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".mp4", ".mov", ".avi", ".mkv"}

def run(project_root: Path, input_path: Path, force: bool = False):
    """
    Copy all supported input files from input_path into project raw/ folder.

    Args:
        project_root (Path): Path to the project root (output folder).
        input_path (Path): Path to the folder containing user input files.
        force (bool): If True, overwrite existing files in raw/.
    """
    project_root = project_root.resolve()
    input_path = input_path.resolve()

    # Ensure project structure ONLY in output/project_root
    paths = ProjectPaths(project_root)
    paths.ensure_all()  # Only creates folders in project_root, not input_path

    logger = get_logger("input", project_root)
    logger.info(f"Starting input stage: copying files from {input_path} into {paths.raw}")

    # Validate input_path
    if not input_path.exists() or not input_path.is_dir():
        raise RuntimeError(f"Input folder does not exist or is not a directory: {input_path}")

    copied_count = 0
    for item in sorted(input_path.iterdir()):
        if item.suffix.lower() in SUPPORTED_EXTS:
            dest = paths.raw / item.name
            if dest.exists() and not force:
                logger.warning(f"Skipping existing file: {item.name}")
                continue
            shutil.copy2(item, dest)
            copied_count += 1

    if copied_count == 0:
        logger.warning("No input files were copied into raw/")
    else:
        logger.info(f"Copied {copied_count} files into raw/: {copied_count} files")
