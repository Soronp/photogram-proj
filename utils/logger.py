import logging
import json
from pathlib import Path
from datetime import datetime


def setup_logger(log_file: Path):
    logger = logging.getLogger("photogrammetry_pipeline")
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Ensure directory exists
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # File handler (UTF-8 FIX)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


# =====================================================
# JSON METRICS LOGGER (NEW)
# =====================================================
def save_metrics_json(output_path: Path, stats: dict, config: dict):
    """
    Save full analysis + config snapshot to JSON.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "timestamp": datetime.now().isoformat(),
        "analysis": stats,
        "final_config": config,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)