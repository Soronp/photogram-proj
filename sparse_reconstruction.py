#!/usr/bin/env python3
"""
sparse_reconstruction.py

MARK-2 Sparse Reconstruction Stage (Canonical)
----------------------------------------------
Responsibilities:
- Run COLMAP mapper
- Deterministically select best sparse model
- Enforce single-sparse invariant
- Emit export_ready.json (authoritative handoff)
- Runner-managed logger
"""

import json
import hashlib
import shutil
import subprocess
from pathlib import Path

from utils.paths import ProjectPaths
from utils.config import load_config, COLMAP_EXE

REQUIRED_FILES = {"cameras.bin", "images.bin", "points3D.bin"}


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def hash_sparse(model_dir: Path) -> str:
    """
    Stable content hash for sparse model identity.
    """
    h = hashlib.sha256()
    for name in sorted(REQUIRED_FILES):
        h.update((model_dir / name).read_bytes())
    return h.hexdigest()


def run_command(cmd, logger, label: str):
    logger.info(f"[sparse] RUN: {label}")
    logger.info(" ".join(map(str, cmd)))

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    if proc.stdout.strip():
        logger.info(proc.stdout)

    if proc.returncode != 0:
        raise RuntimeError(f"[sparse] {label} failed")


# --------------------------------------------------
# Pipeline Stage
# --------------------------------------------------
def run(run_root: Path, project_root: Path, force: bool, logger):
    project_root = project_root.resolve()

    # Run-scoped immutable config
    load_config(run_root)

    paths = ProjectPaths(project_root)
    paths.ensure_all()

    logger.info("[sparse] Stage started")

    # --------------------------------------------------
    # Reset sparse directory (force-aware)
    # --------------------------------------------------
    if paths.sparse.exists() and force:
        logger.info("[sparse] Force enabled — removing existing sparse/")
        shutil.rmtree(paths.sparse)

    paths.sparse.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # COLMAP mapper
    # --------------------------------------------------
    run_command(
        [
            COLMAP_EXE, "mapper",
            "--database_path", paths.database / "database.db",
            "--image_path", paths.images_processed,
            "--output_path", paths.sparse,
        ],
        logger,
        label="COLMAP Mapper",
    )

    # --------------------------------------------------
    # Collect valid sparse models
    # --------------------------------------------------
    models = sorted(
        d for d in paths.sparse.iterdir()
        if d.is_dir() and REQUIRED_FILES.issubset({f.name for f in d.iterdir()})
    )

    if not models:
        raise RuntimeError("[sparse] No valid sparse models produced")

    # --------------------------------------------------
    # Deterministic selection
    # Strategy: largest points3D.bin
    # --------------------------------------------------
    best_model = max(
        models,
        key=lambda d: (d / "points3D.bin").stat().st_size,
    )

    logger.info(f"[sparse] Selected model: {best_model.name}")

    # --------------------------------------------------
    # Enforce single-model invariant
    # --------------------------------------------------
    for d in models:
        if d != best_model:
            shutil.rmtree(d)

    # --------------------------------------------------
    # Authoritative export metadata
    # --------------------------------------------------
    meta = {
        "model_dir": best_model.name,
        "format": "COLMAP",
        "sparse_hash": hash_sparse(best_model),
        "ready_for_openmvs": True,
    }

    meta_path = paths.sparse / "export_ready.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    logger.info("[sparse] export_ready.json written")
    logger.info("[sparse] Stage completed successfully")
