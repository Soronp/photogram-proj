import logging
import os
from datetime import datetime
from utils.config import PATHS

_LOGGER = None


def get_logger(name: str = "MARK2"):
    """
    Returns a singleton logger instance.
    Safe to call from any module.
    """
    global _LOGGER
    if _LOGGER is not None:
        return _LOGGER

    os.makedirs(PATHS["logs"], exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(PATHS["logs"], f"pipeline_{timestamp}.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s"
    )

    # File handler
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(formatter)

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info("Logger initialized")
    logger.info(f"Log file: {log_file}")

    _LOGGER = logger
    return logger
