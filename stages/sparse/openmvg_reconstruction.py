from pathlib import Path
from core.tool_runner import ToolRunner
from utils.paths import ProjectPaths
from config.config_manager import load_config
import shutil


def run(run_root: Path, project_root: Path, force: bool, logger):
    stage = "openmvg_reconstruction"

    # --------------------------------------------------
    # INIT
    # --------------------------------------------------
    paths = ProjectPaths(project_root, run_root.name)
    tool = ToolRunner(logger)

    config = load_config()
    cfg = config.get("sparse", {}).get("openmvg", {})

    input_images = paths.images
    sparse_root = paths.sparse

    matches_dir = sparse_root / "openmvg_matches"
    reconstruction_dir = sparse_root / "openmvg_reconstruction"

    sensor_db = cfg.get("sensor_database")  # MUST be set in config

    # --------------------------------------------------
    # CLEAN START
    # --------------------------------------------------
    if matches_dir.exists():
        shutil.rmtree(matches_dir)
    if reconstruction_dir.exists():
        shutil.rmtree(reconstruction_dir)

    matches_dir.mkdir(parents=True, exist_ok=True)
    reconstruction_dir.mkdir(parents=True, exist_ok=True)

    sfm_output = reconstruction_dir / "sfm_data.bin"

    if sfm_output.exists() and not force:
        logger.info(f"[{stage}] Skipping (already exists)")
        return {"sfm_data": sfm_output}

    logger.info(f"[{stage}] Starting OpenMVG pipeline")

    # --------------------------------------------------
    # AUTO FOCAL LENGTH
    # --------------------------------------------------
    try:
        from PIL import Image
        first_image = next(input_images.glob("*.*"))
        with Image.open(first_image) as img:
            width, height = img.size
        focal_length = max(width, height)
    except Exception:
        focal_length = 2400

    logger.info(f"[{stage}] Using focal length: {focal_length}")

    # --------------------------------------------------
    # 1. IMAGE LISTING
    # --------------------------------------------------
    cmd = [
        "openMVG_main_SfMInit_ImageListing",
        "-i", str(input_images),
        "-o", str(matches_dir),
        "-c", "1",
        "-f", str(focal_length),
    ]

    if sensor_db:
        cmd += ["-d", str(sensor_db)]

    tool.run(cmd, stage=stage)

    sfm_json = matches_dir / "sfm_data.json"

    if not sfm_json.exists():
        raise RuntimeError(f"[{stage}] sfm_data.json not created")

    logger.info(f"[{stage}] sfm_data.json located at: {sfm_json}")

    # --------------------------------------------------
    # 2. FEATURE EXTRACTION
    # --------------------------------------------------
    tool.run(
        [
            "openMVG_main_ComputeFeatures",
            "-i", str(sfm_json),
            "-o", str(matches_dir),
            "-m", "SIFT",
            "-p", "NORMAL",
            "-f", "1"
        ],
        stage=stage
    )

    # --------------------------------------------------
    # VERIFY FEATURES
    # --------------------------------------------------
    feat_files = list(matches_dir.glob("*.feat"))
    desc_files = list(matches_dir.glob("*.desc"))

    logger.info(f"[{stage}] Found {len(feat_files)} .feat and {len(desc_files)} .desc files")

    if len(feat_files) == 0 or len(desc_files) == 0:
        raise RuntimeError(f"[{stage}] Feature extraction failed")

    # --------------------------------------------------
    # 3. FEATURE MATCHING
    # --------------------------------------------------
    tool.run(
        [
            "openMVG_main_ComputeMatches",
            "-i", str(sfm_json),
            "-o", str(matches_dir),
            "-n", "FASTCASCADEHASHINGL2",
            "-f", "1"
        ],
        stage=stage
    )

    # OPTIONAL: check matches file
    matches_bin = matches_dir / "matches.f.bin"
    matches_txt = matches_dir / "matches.f.txt"

    if not matches_bin.exists() and not matches_txt.exists():
        raise RuntimeError(f"[{stage}] Matching failed (no matches file)")

    # --------------------------------------------------
    # 4. INCREMENTAL SFM
    # --------------------------------------------------
    tool.run(
        [
            "openMVG_main_IncrementalSfM",
            "-i", str(sfm_json),
            "-m", str(matches_dir),
            "-o", str(reconstruction_dir)
        ],
        stage=stage
    )

    # --------------------------------------------------
    # VALIDATION
    # --------------------------------------------------
    if not sfm_output.exists():
        raise RuntimeError(f"[{stage}] Reconstruction failed (no sfm_data.bin)")

    logger.info(f"[{stage}] DONE")

    return {
        "sfm_data": sfm_output,
        "matches_dir": matches_dir,
        "reconstruction_dir": reconstruction_dir
    }