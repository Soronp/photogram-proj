import logging
import json
from pathlib import Path
from datetime import datetime
import uuid


class MetricsCollector:
    def __init__(self):
        self.data = {}
        self.run_id = str(uuid.uuid4())

    def log(self, stage, values: dict):
        """
        Store per-stage metrics.
        Supports multiple entries per stage (for retries).
        """
        if stage not in self.data:
            self.data[stage] = []

        self.data[stage].append(values)

    def export(self):
        return {
            "run_id": self.run_id,
            "stages": self.data
        }


def setup_logger(log_file: Path):
    logger = logging.getLogger("photogrammetry_pipeline")
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    log_file.parent.mkdir(parents=True, exist_ok=True)

    # File logging
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # Console logging
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    # Attach metrics collector
    logger.metrics = MetricsCollector()

    return logger


# =====================================================
# JSON EXPORT
# =====================================================
def save_metrics_json(output_path: Path, stats: dict, config: dict, logger=None):

    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "timestamp": datetime.now().isoformat(),
        "analysis": stats,
        "final_config": config,
    }

    if logger and hasattr(logger, "metrics"):
        payload["metrics"] = logger.metrics.export()

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)