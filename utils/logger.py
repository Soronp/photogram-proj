import logging
from pathlib import Path


def get_run_logger(
    run_id: str,
    run_logs_dir: Path,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    MARK-2 authoritative run-scoped logger.

    Rules:
    - Exactly ONE log file per run
    - Location: <project_root>/runs/<run_id>/logs/run.log
    - Logger must be initialized ONCE by the runner
    - Pipeline stages may only retrieve it by name

    Args:
        run_id: Run identifier (from RunContext)
        run_logs_dir: RunContext.logs directory
        level: Logging level
    """

    run_logs_dir = run_logs_dir.resolve()
    run_logs_dir.mkdir(parents=True, exist_ok=True)

    logger_name = f"MARK-2::{run_id}"
    logger = logging.getLogger(logger_name)

    if logger.handlers:
        return logger  # already initialized

    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler — SINGLE run log
    file_handler = logging.FileHandler(run_logs_dir / "run.log", encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
