#!/usr/bin/env python3
"""
sparse_reconstruction.py

MARK-2 Sparse Reconstruction Stage (Canonical)
----------------------------------------------
- ToolRunner enforced
- Deterministic model selection
- Single sparse invariant
- Immutable config
"""

import json
import hashlib
import shutil
from pathlib import Path

from utils.paths import ProjectPaths
from config_manager import load_config
from tool_runner import ToolRunner

REQUIRED_FILES = {"cameras.bin", "images.bin", "points3D.bin"}


def _hash_sparse(model_dir: Path) -> str:
    h = hashlib.sha256()
    for name in sorted(REQUIRED_FILES):
        h.update((model_dir / name).read_bytes())
    return h.hexdigest()


def run(run_root: Path, project_root: Path, force: bool, logger):
    paths = ProjectPaths(project_root)
    paths.ensure_all()

    logger.info("[sparse] START")

    config = load_config(run_root, logger)
    tool = ToolRunner(config, logger)

    if paths.sparse.exists() and force:
        shutil.rmtree(paths.sparse)

    paths.sparse.mkdir(parents=True, exist_ok=True)

    tool.run(
        "colmap",
        [
            "mapper",
            "--database_path", paths.database / "database.db",
            "--image_path", paths.images_processed,
            "--output_path", paths.sparse,
        ],
    )

    models = [
        d for d in paths.sparse.iterdir()
        if d.is_dir() and REQUIRED_FILES.issubset({f.name for f in d.iterdir()})
    ]

    if not models:
        raise RuntimeError("[sparse] No valid sparse models produced")

    best = max(models, key=lambda d: (d / "points3D.bin").stat().st_size)

    for d in models:
        if d != best:
            shutil.rmtree(d)

    meta = {
        "model_dir": best.name,
        "format": "COLMAP",
        "sparse_hash": _hash_sparse(best),
        "ready_for_openmvs": True,
    }

    (paths.sparse / "export_ready.json").write_text(json.dumps(meta, indent=2))

    logger.info("[sparse] COMPLETED")
