#!/usr/bin/env python3
"""
minimal_monitor_hms.py

MARK-2 Pipeline Monitor (Minimal)
- Progress bar for latest stage
- Accurate elapsed time (HH:MM:SS) based on log file timestamps
- Prompts for logs directory
"""

import time
from pathlib import Path
import re

PIPELINE_STAGES = [
    'init', 'ingest', 'image_filter', 'image_analyzer', 'pre_processing',
    'database_builder', 'matcher', 'sparse_reconstruction', 'sparse_evaluation',
    'dense_reconstruction', 'dense_evaluation', 'gen_mesh', 'mesh_postprocess',
    'texture_mapping', 'mesh_evaluation', 'evaluation_aggregator', 'visualization'
]


def ask_log_directory():
    while True:
        path = input("Enter path to logs directory: ").strip().strip('"')
        log_dir = Path(path)
        if log_dir.exists() and log_dir.is_dir():
            return log_dir
        print(f"Directory not found: {path}")


def latest_stage(log_dir: Path):
    latest = None
    latest_mtime = 0
    for stage in PIPELINE_STAGES:
        log_file = log_dir / f"{stage}.log"
        if log_file.exists() and log_file.stat().st_mtime > latest_mtime:
            latest_mtime = log_file.stat().st_mtime
            latest = stage
    return latest


def first_stage_time(log_dir: Path):
    earliest_time = None
    for stage in PIPELINE_STAGES:
        log_file = log_dir / f"{stage}.log"
        if log_file.exists():
            mtime = log_file.stat().st_mtime
            if earliest_time is None or mtime < earliest_time:
                earliest_time = mtime
    return earliest_time


def extract_progress(log_file: Path):
    if not log_file.exists():
        return 0
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        # X/Y pattern
        matches = re.findall(r'(\d+)\s*/\s*(\d+)', content)
        if matches:
            current, total = map(int, matches[-1])
            return (current / total) * 100 if total > 0 else 0
        # Percentage pattern
        matches = re.findall(r'(\d+)%', content)
        if matches:
            return float(matches[-1])
    except:
        return 0
    return 0


def format_elapsed(seconds):
    """Format seconds as HH:MM:SS"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def display_bar(stage, progress, elapsed):
    bar_width = 30
    filled = int(bar_width * progress / 100)
    bar = "#" * filled + "-" * (bar_width - filled)
    print(f"\r[{stage}] [{bar}] {progress:.1f}% | Elapsed: {format_elapsed(elapsed)}", end="", flush=True)


def monitor():
    log_dir = ask_log_directory()
    print(f"Monitoring logs in: {log_dir}\nPress Ctrl+C to stop.\n")

    try:
        while True:
            stage = latest_stage(log_dir)
            if stage is None:
                print("\r[WAITING] No logs detected yet...", end="", flush=True)
                time.sleep(2)
                continue

            log_file = log_dir / f"{stage}.log"
            progress = extract_progress(log_file)

            start_time = first_stage_time(log_dir)
            if start_time is not None:
                latest_time = log_file.stat().st_mtime
                elapsed = latest_time - start_time
            else:
                elapsed = 0

            display_bar(stage, progress, elapsed)
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nMonitoring stopped by user.")


if __name__ == "__main__":
    monitor()
