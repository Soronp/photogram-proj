#!/usr/bin/env python3
"""
sparse_reconstruction.py

MARK-2 Sparse Reconstruction (Authoritative)
---------------------------------------------
- Runs COLMAP mapper
- Selects best sparse model deterministically
- Physically enforces ONE active sparse model
- Writes export_ready.json (single source of truth)
- Runner-managed logger
"""

import json
import hashlib
import shutil
import subprocess
from pathlib import Path

from utils.paths import ProjectPaths
from utils.config import load_config, COLMAP_EXE


REQUIRED = {"cameras.bin", "images.bin", "points3D.bin"}


def hash_sparse(model_dir: Path) -> str:
    h = hashlib.sha256()
    for name in sorted(REQUIRED):
        h.update((model_dir / name).read_bytes())
    return h.hexdigest()


def run_sparse(run_root: Path, project_root: Path, force: bool, logger):
    project_root = Path(project_root).resolve()
    load_config(project_root)

    paths = ProjectPaths(project_root)
    paths.ensure_all()

    logger.info("[sparse] Starting sparse reconstruction")

    if paths.sparse.exists() and force:
        logger.info("[sparse] Force enabled — removing existing sparse/")
        shutil.rmtree(paths.sparse)

    paths.sparse.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # COLMAP mapper
    # --------------------------------------------------
    cmd = [
        COLMAP_EXE, "mapper",
        "--database_path", paths.database / "database.db",
        "--image_path", paths.images_processed,
        "--output_path", paths.sparse,
    ]

    logger.info("[sparse] Running COLMAP mapper")
    subprocess.run(cmd, check=True)

    # --------------------------------------------------
    # Collect valid sparse models
    # --------------------------------------------------
    models = []
    for d in sorted(paths.sparse.iterdir()):
        if d.is_dir() and REQUIRED.issubset({f.name for f in d.iterdir()}):
            models.append(d)

    if not models:
        raise RuntimeError("No valid sparse models produced")

    # --------------------------------------------------
    # Select best model (largest points3D.bin)
    # --------------------------------------------------
    best = max(models, key=lambda d: (d / "points3D.bin").stat().st_size)
    logger.info(f"[sparse] Selected model: {best.name}")

    # --------------------------------------------------
    # Enforce single-sparse invariant
    # --------------------------------------------------
    for d in models:
        if d != best:
            shutil.rmtree(d)

    # --------------------------------------------------
    # Write authoritative metadata
    # --------------------------------------------------
    meta = {
        "model_dir": best.name,
        "format": "COLMAP",
        "sparse_hash": hash_sparse(best),
        "ready_for_openmvs": True,
    }

    meta_path = paths.sparse / "export_ready.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    logger.info("[sparse] Sparse reconstruction finalized")
