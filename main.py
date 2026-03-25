from pathlib import Path
import shutil

from core.runner import PipelineRunner
from config.config_manager import load_config

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def is_valid_image(file: Path):
    return file.is_file() and file.suffix.lower() in VALID_EXT


# =====================================================
# INPUT
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

    if raw_images_dir.exists():
        shutil.rmtree(raw_images_dir)

    raw_images_dir.mkdir(parents=True, exist_ok=True)

    print("\nCopying images...")
    for img in input_path.iterdir():
        if is_valid_image(img):
            shutil.copy2(img, raw_images_dir / img.name)

    return project_root, raw_images_dir


# =====================================================
# PIPELINE CHOICE
# =====================================================
def get_pipeline_choice():
    print("\n=== Pipeline Selection ===")
    print("A → COLMAP full")
    print("B → GLOMAP + COLMAP dense")
    print("C → COLMAP + OpenMVS (full MVS pipeline)")
    print("D → OpenMVG (SfM only)")  # 🔥 NEW

    choice = input("Select pipeline [A]: ").strip().upper()
    return choice if choice in ["A", "B", "C", "D"] else "A"


# =====================================================
# CONFIG
# =====================================================
def get_user_config(project_root: Path, image_source: Path, pipeline_choice: str):

    # 🔥 Updated backend mapping
    backend_map = {
        "A": "colmap",
        "B": "glomap",
        "C": "colmap",
        "D": "openmvg"   # NEW
    }

    # Base config
    user_config = {
        "paths": {"project_root": str(project_root)},
        "ingestion": {"external_image_path": str(image_source)},
        "downsampling": {"enabled": True},
        "sparse": {"backend": backend_map[pipeline_choice]},
    }

    # 🔥 Pipeline-specific overrides
    if pipeline_choice == "C":
        user_config.setdefault("pipeline", {})
        user_config["pipeline"].update({
            "dense_backend": "openmvs",
            "mesh_backend": "openmvs",
            "texture_backend": "openmvs",
        })

    if pipeline_choice == "D":
        # OpenMVG = sparse only (for now)
        user_config.setdefault("pipeline", {})
        user_config["pipeline"].update({
            "dense_backend": None,
            "mesh_backend": None,
            "texture_backend": None,
        })

    return user_config


# =====================================================
# MAIN
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