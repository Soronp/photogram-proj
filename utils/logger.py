# utils/logger.py
import logging
from pathlib import Path
from datetime import datetime


def get_logger(name: str, project_root: Path, level=logging.INFO) -> logging.Logger:
    """
    Centralized logger factory.

    Creates:
      logs/<name>.log

    Features:
    - Timestamped logs
    - Console + file output
    - Safe to call multiple times (no duplicate handlers)
    """
    logs_dir = project_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger  # Prevent duplicate handlers

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler
    log_file = logs_dir / f"{name}.log"
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    fh.setLevel(level)

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    ch.setLevel(level)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info("Logger initialized")
    return logger


