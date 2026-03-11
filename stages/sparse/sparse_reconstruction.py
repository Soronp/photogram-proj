#!/usr/bin/env python3
"""
sparse_reconstruction.py

Stage 6 — Sparse reconstruction

Responsibilities
----------------
• run COLMAP mapper
• identify produced models
• select best model
• export metadata for dense stage
"""

import json
import hashlib


REQUIRED_FILES = {"cameras.bin", "images.bin", "points3D.bin"}


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def hash_sparse_model(model_dir):

    h = hashlib.sha256()

    for name in REQUIRED_FILES:

        f = model_dir / name

        if f.exists():
            h.update(f.read_bytes())

    return h.hexdigest()


def collect_images(images_dir):

    return [
        p for p in images_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]


def choose_mapper(n):

    if n >= 2000:
        return "hierarchical_mapper"

    return "mapper"


# --------------------------------------------------
# Stage execution
# --------------------------------------------------

def run(paths, logger, tools, config):

    logger.info("[sparse] stage started")

    images_dir = paths.images_filtered
    sparse_dir = paths.sparse
    db = paths.database_path

    if not db.exists():
        raise RuntimeError("database.db missing")

    sparse_dir.mkdir(parents=True, exist_ok=True)

    images = collect_images(images_dir)

    if not images:
        raise RuntimeError("no images available for reconstruction")

    n_images = len(images)

    logger.info(f"[sparse] images: {n_images}")

    mapper = choose_mapper(n_images)

    logger.info(f"[sparse] mapper: {mapper}")

    mapper_cfg = config.get("mapper", {})

    tools.run(
        "colmap",
        [
            mapper,

            "--database_path", str(db),
            "--image_path", str(images_dir),
            "--output_path", str(sparse_dir),

            "--Mapper.num_threads", "-1",

            "--Mapper.init_min_num_inliers",
            str(mapper_cfg.get("init_min_num_inliers", 80)),

            "--Mapper.init_max_error",
            str(mapper_cfg.get("init_max_error", 4)),

            "--Mapper.filter_max_reproj_error",
            str(mapper_cfg.get("filter_max_reproj_error", 4)),

            "--Mapper.filter_min_tri_angle",
            str(mapper_cfg.get("filter_min_tri_angle", 2)),

            "--Mapper.tri_ignore_two_view_tracks", "0",

            "--Mapper.ba_local_max_num_iterations",
            str(mapper_cfg.get("ba_local_iterations", 40)),

            "--Mapper.ba_global_max_num_iterations",
            str(mapper_cfg.get("ba_global_iterations", 80)),

            "--Mapper.ba_refine_focal_length", "1",
            "--Mapper.ba_refine_principal_point", "0",
            "--Mapper.ba_refine_extra_params", "1",
        ],
    )

    # --------------------------------------------------
    # Identify produced sparse models
    # --------------------------------------------------

    models = []

    for d in sparse_dir.iterdir():

        if not d.is_dir():
            continue

        files = {f.name for f in d.iterdir()}

        if REQUIRED_FILES.issubset(files):
            models.append(d)

    if not models:
        raise RuntimeError("COLMAP produced no sparse models")

    best = max(models, key=lambda m: (m / "points3D.bin").stat().st_size)

    logger.info(f"[sparse] best model: {best.name}")

    meta = {
        "model_dir": best.name,
        "sparse_hash": hash_sparse_model(best),
        "ready_for_dense": True
    }

    meta_path = sparse_dir / "export_ready.json"

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("[sparse] stage completed")