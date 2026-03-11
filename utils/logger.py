#!/usr/bin/env python3
"""
logger.py

Run logger utilities for the MARK-2 pipeline.
"""

import logging
from pathlib import Path


# --------------------------------------------------
# Logger Creation
# --------------------------------------------------

def create_run_logger(run_id: str, logs_dir: Path) -> logging.Logger:
    """
    Create the canonical run logger.

    Output:
        workspace/runs/<run_id>/logs/run.log
    """

    logs_dir = Path(logs_dir).resolve()
    logs_dir.mkdir(parents=True, exist_ok=True)

    logger_name = f"pipeline::{run_id}"

    logger = logging.getLogger(logger_name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )

    # -------------------------
    # File handler
    # -------------------------

    file_path = logs_dir / "run.log"

    file_handler = logging.FileHandler(file_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    # -------------------------
    # Console handler
    # -------------------------

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info("Logger initialized")

    return logger


# --------------------------------------------------
# Retrieve existing logger
# --------------------------------------------------

def get_logger(run_id: str) -> logging.Logger:

    logger_name = f"pipeline::{run_id}"

    return logging.getLogger(logger_name)