import logging
from pathlib import Path


def get_logger(name: str, log_root: Path, level=logging.INFO) -> logging.Logger:
    """
    Centralized logger factory (run/output aware).

    Args:
        name (str): Logger name (stage name)
        log_root (Path): Directory where logs will be written
        level: Logging level

    Creates:
        <log_root>/<name>.log

    Features:
    - Timestamped logs
    - Console + file output
    - Safe to call multiple times (no duplicate handlers)
    """

    log_root = log_root.resolve()
    log_root.mkdir(parents=True, exist_ok=True)

    logger_id = f"{name}@{log_root}"
    logger = logging.getLogger(logger_id)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler
    file_handler = logging.FileHandler(log_root / f"{name}.log")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
