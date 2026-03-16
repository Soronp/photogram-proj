import logging
from pathlib import Path


def setup_logger(log_dir: Path, level=logging.INFO):
    """
    Initialize a unified pipeline logger.

    Parameters
    ----------
    log_dir : Path
        Directory where logs will be written.
    level : logging level
        Default logging level.
    """

    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "pipeline.log"

    logger = logging.getLogger("photogrammetry_pipeline")
    logger.setLevel(level)

    if logger.hasHandlers():
        logger.handlers.clear()

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        "%Y-%m-%d %H:%M:%S"
    )

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger