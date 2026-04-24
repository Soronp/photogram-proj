from pathlib import Path
import shutil

from core.runner import PipelineRunner
from config.config_manager import load_config


# =====================================================
# CONSTANTS
# =====================================================
VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def is_valid_image(file: Path):
    return file.is_file() and file.suffix.lower() in VALID_EXT


# =====================================================
# INPUT HANDLING
# =====================================================
def get_user_paths():
    print("=== Photogrammetry Pipeline Setup ===")

    input_path = Path(input("Enter path to IMAGE folder: ").strip())
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    project_root = Path(input("Enter OUTPUT project folder: ").strip())
    raw_images_dir = project_root / "raw_images"

    copy_choice = input("Copy images into project? (y/n) [y]: ").strip().lower()
    copy_enabled = (copy_choice != "n")

    if not copy_enabled:
        return project_root, input_path

    # Clean existing
    if raw_images_dir.exists():
        shutil.rmtree(raw_images_dir)

    raw_images_dir.mkdir(parents=True, exist_ok=True)

    print("\nCopying images...")
    count = 0
    for img in input_path.iterdir():
        if is_valid_image(img):
            shutil.copy2(img, raw_images_dir / img.name)
            count += 1

    print(f"Copied {count} images")

    return project_root, raw_images_dir


# =====================================================
# PIPELINE SELECTION
# =====================================================
def get_pipeline_choice():
    print("\n=== Pipeline Selection ===")
    print("A → COLMAP full")
    print("B → GLOMAP + COLMAP dense")
    print("C → COLMAP + OpenMVS")
    print("D → OpenMVG (SfM only)")
    print("E → COLMAP + Nerfstudio (Neural Reconstruction)")

    choice = input("Select pipeline [A]: ").strip().upper()
    return choice if choice in ["A", "B", "C", "D", "E"] else "A"


# =====================================================
# CONFIG BUILDER
# =====================================================
def get_user_config(project_root: Path, image_source: Path, pipeline_choice: str):

    user_config = {
        "paths": {"project_root": str(project_root)},
        "ingestion": {"external_image_path": str(image_source)},
        "downsampling": {"enabled": True},
    }

    # -------------------------------------------------
    # PIPELINE-SPECIFIC CONFIG
    # -------------------------------------------------
    if pipeline_choice == "C":
        user_config.setdefault("pipeline", {})
        user_config["pipeline"]["backends"] = {
            "sparse": "colmap",
            "dense": "openmvs",
            "mesh": "openmvs",
            "texture": "openmvs"
        }

    elif pipeline_choice == "D":
        user_config.setdefault("pipeline", {})
        user_config["pipeline"]["backends"] = {
            "sparse": "openmvg",
            "dense": None,
            "mesh": None,
            "texture": None
        }

    elif pipeline_choice == "E":
        # 🔥 NERFSTUDIO PIPELINE
        user_config.setdefault("pipeline", {})
        user_config["pipeline"]["backends"] = {
            "sparse": "colmap",
            "dense": "nerfstudio",   # ✅ CORRECT
            "mesh": "colmap",        # optional mesh post-process
            "texture": None
        }

        # 🔥 Nerfstudio config (aligned with your new config system)
        user_config.setdefault("dense", {})
        user_config["dense"]["nerfstudio"] = {
            "method": "splatfacto",
            "iterations": 15000,   # faster for testing
            "use_gpu": True,
            "use_downsampled": True,
            "export": {
                "type": "pointcloud",
                "resolution": 512
            }
        }

    return user_config


# =====================================================
# MAIN ENTRY
# =====================================================
if __name__ == "__main__":
    try:
        project_root, image_source = get_user_paths()
        pipeline_choice = get_pipeline_choice()

        user_config = get_user_config(project_root, image_source, pipeline_choice)
        config = load_config(user_config)

        runner = PipelineRunner(config, pipeline_type=pipeline_choice)
        runner.run()

        print("\n✅ Pipeline completed successfully!")

    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")