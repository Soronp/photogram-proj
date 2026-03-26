from pathlib import Path
from core.tool_runner import ToolRunner
from utils.paths import ProjectPaths
from config.config_manager import load_config

import shutil
import json
from PIL import Image


def run(run_root: Path, project_root: Path, force: bool, logger):
    stage = "openmvg_reconstruction"

    paths = ProjectPaths(project_root, run_root.name)
    tool = ToolRunner(logger)

    config = load_config()
    cfg = config.get("sparse", {}).get("openmvg", {})

    input_images = paths.images
    sparse_root = paths.sparse

    matches_dir = sparse_root / "openmvg_matches"
    reconstruction_dir = sparse_root / "openmvg_reconstruction"

    sensor_db = cfg.get("sensor_database")
    focal_multiplier = cfg.get("fallback_focal_multiplier", 1.2)

    # --------------------------------------------------
    # CLEAN START
    # --------------------------------------------------
    if matches_dir.exists():
        shutil.rmtree(matches_dir)
    if reconstruction_dir.exists():
        shutil.rmtree(reconstruction_dir)

    matches_dir.mkdir(parents=True, exist_ok=True)
    reconstruction_dir.mkdir(parents=True, exist_ok=True)

    sfm_json = matches_dir / "sfm_data.json"
    sfm_bin = reconstruction_dir / "sfm_data.bin"

    logger.info(f"[{stage}] Starting OpenMVG pipeline")

    # --------------------------------------------------
    # PREPARE FALLBACK FOCAL
    # --------------------------------------------------
    first_image = next(input_images.glob("*.*"))
    with Image.open(first_image) as img:
        width, height = img.size

    fallback_focal = focal_multiplier * max(width, height)

    # --------------------------------------------------
    # 1. IMAGE LISTING (ROBUST INTRINSICS)
    # --------------------------------------------------
    def run_listing(use_sensor_db=True):
        cmd = [
            "openMVG_main_SfMInit_ImageListing",
            "-i", str(input_images),
            "-o", str(matches_dir),
            "-c", "3",
        ]

        if use_sensor_db and sensor_db:
            cmd += ["-d", str(sensor_db)]
            logger.info(f"[{stage}] Using sensor DB")
        else:
            cmd += ["-f", str(fallback_focal)]
            logger.warning(f"[{stage}] Using fallback focal: {fallback_focal}")

        tool.run(cmd, stage=stage)

    run_listing(True)

    if not sfm_json.exists():
        raise RuntimeError(f"[{stage}] sfm_data.json not created")

    with open(sfm_json) as f:
        data = json.load(f)

    if len(data.get("intrinsics", [])) == 0:
        logger.warning(f"[{stage}] Sensor DB failed → fallback")

        shutil.rmtree(matches_dir)
        matches_dir.mkdir(parents=True, exist_ok=True)

        run_listing(False)

        with open(sfm_json) as f:
            data = json.load(f)

        if len(data.get("intrinsics", [])) == 0:
            raise RuntimeError(f"[{stage}] ❌ Intrinsics failed completely")

    logger.info(f"[{stage}] ✅ Intrinsics OK")

    # --------------------------------------------------
    # 2. FEATURE EXTRACTION (MAX DETAIL)
    # --------------------------------------------------
    tool.run(
        [
            "openMVG_main_ComputeFeatures",
            "-i", str(sfm_json),
            "-o", str(matches_dir),
            "-m", "SIFT",
            "-p", "ULTRA",   # 🔥 max detail
            "-f", "1"
        ],
        stage=stage
    )

    # --------------------------------------------------
    # 3. FEATURE MATCHING (MAX PERMISSIVE)
    # --------------------------------------------------
    tool.run(
        [
            "openMVG_main_ComputeMatches",
            "-i", str(sfm_json),
            "-o", str(matches_dir),
            "-n", "ANNL2",
            "-g", "f",   # 🔥 fundamental (more permissive)
            "--guided_matching", "1",
            "--ratio", "0.95",
            "-f", "1"
        ],
        stage=stage
    )

    # --------------------------------------------------
    # MATCH DETECTION
    # --------------------------------------------------
    match_candidates = [
        matches_dir / "matches.f.bin",
        matches_dir / "matches.e.bin",
        matches_dir / "matches.h.bin",
        matches_dir / "matches.f.txt",
        matches_dir / "matches.e.txt",
        matches_dir / "matches.h.txt",
    ]

    match_file = next((m for m in match_candidates if m.exists()), None)

    if match_file is None:
        raise RuntimeError(f"[{stage}] ❌ No match file found")

    if match_file.stat().st_size < 2000:
        raise RuntimeError(f"[{stage}] ❌ Matches too weak")

    logger.info(f"[{stage}] ✅ Matches OK → {match_file.name}")

    # --------------------------------------------------
    # 4. INCREMENTAL SFM FIRST (KEY CHANGE)
    # --------------------------------------------------
    logger.info(f"[{stage}] Running Incremental SfM (primary)")

    tool.run(
        [
            "openMVG_main_IncrementalSfM",
            "-i", str(sfm_json),
            "-m", str(matches_dir),
            "-o", str(reconstruction_dir)
        ],
        stage=stage
    )

    def is_valid():
        return sfm_bin.exists() and sfm_bin.stat().st_size > 20000

    # --------------------------------------------------
    # 5. FALLBACK TO GLOBAL
    # --------------------------------------------------
    if not is_valid():
        logger.warning(f"[{stage}] Incremental weak → trying Global")

        shutil.rmtree(reconstruction_dir)
        reconstruction_dir.mkdir(parents=True, exist_ok=True)

        tool.run(
            [
                "openMVG_main_GlobalSfM",
                "-i", str(sfm_json),
                "-m", str(matches_dir),
                "-o", str(reconstruction_dir),
                "-r", "1",
                "-t", "3",
                "-f", "ADJUST_FOCAL_LENGTH"
            ],
            stage=stage
        )

    if not is_valid():
        raise RuntimeError(f"[{stage}] ❌ SfM failed completely")

    logger.info(f"[{stage}] ✅ SfM reconstruction OK")

    # --------------------------------------------------
    # 6. MAX STRUCTURE DENSIFICATION
    # --------------------------------------------------
    dense_bin = reconstruction_dir / "dense.bin"

    tool.run(
        [
            "openMVG_main_ComputeStructureFromKnownPoses",
            "-i", str(sfm_bin),
            "-m", str(matches_dir),
            "-o", str(dense_bin)
        ],
        stage=stage
    )

    if dense_bin.exists():
        sfm_bin = dense_bin
        logger.info(f"[{stage}] Using dense structure")

    # --------------------------------------------------
    # 7. EXPORT TO OPENMVS
    # --------------------------------------------------
    mvs_dir = reconstruction_dir / "openmvs"
    mvs_dir.mkdir(exist_ok=True)

    tool.run(
        [
            "openMVG_main_openMVG2openMVS",
            "-i", str(sfm_bin),
            "-o", str(mvs_dir / "scene.mvs"),
            "-d", str(mvs_dir)
        ],
        stage=stage
    )

    mvs_scene = mvs_dir / "scene.mvs"

    if not mvs_scene.exists():
        raise RuntimeError(f"[{stage}] ❌ OpenMVS export failed")

    logger.info(f"[{stage}] ✅ OpenMVS export OK")
    logger.info(f"[{stage}] DONE")

    return {
        "sfm_data": sfm_bin,
        "matches_dir": matches_dir,
        "reconstruction_dir": reconstruction_dir,
        "mvs_scene": mvs_scene,
        "match_file_used": match_file.name
    }