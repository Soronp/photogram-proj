#!/usr/bin/env python3
"""
sparse_reconstruction.py

MARK-2 Sparse Reconstruction
----------------------------
- Runs COLMAP mapper
- Selects best sparse model deterministically
- Writes export_ready.json with sparse hash
"""

import json
import hashlib
import shutil
import subprocess
from pathlib import Path

from utils.logger import get_logger
from utils.paths import ProjectPaths
from utils.config import load_config, COLMAP_EXE


def hash_sparse(model_dir: Path) -> str:
    h = hashlib.sha256()
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        p = model_dir / name
        h.update(p.read_bytes())
    return h.hexdigest()


def run_sparse(project_root: Path, force: bool):
    load_config(project_root)
    paths = ProjectPaths(project_root)
    paths.ensure_all()

    logger = get_logger("sparse_reconstruction", project_root)

    if paths.sparse.exists() and force:
        shutil.rmtree(paths.sparse)

    paths.sparse.mkdir(parents=True, exist_ok=True)

    cmd = [
        COLMAP_EXE, "mapper",
        "--database_path", paths.database / "database.db",
        "--image_path", paths.images_processed,
        "--output_path", paths.sparse,
    ]

    logger.info("[RUN:COLMAP mapper]")
    subprocess.run(cmd, check=True)

    candidates = [d for d in paths.sparse.iterdir() if d.is_dir()]
    if not candidates:
        raise RuntimeError("No sparse models produced")

    best = max(
        candidates,
        key=lambda d: (d / "points3D.bin").stat().st_size,
    )

    sparse_hash = hash_sparse(best)

    meta = {
        "model_dir": best.name,
        "format": "COLMAP",
        "sparse_hash": sparse_hash,
        "ready_for_openmvs": True,
    }

    (paths.sparse / "export_ready.json").write_text(
        json.dumps(meta, indent=2)
    )

    logger.info(f"Selected sparse model: {best.name}")
