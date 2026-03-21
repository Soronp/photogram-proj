from pathlib import Path
import shutil

from core.runner import PipelineRunner
from config.config_manager import load_config


# =====================================================
# IMAGE VALIDATION
# =====================================================
VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def is_valid_image(file: Path):
    return file.is_file() and file.suffix.lower() in VALID_EXT


# =====================================================
# USER INPUT
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
        print("Using images in-place (advanced mode)")
        return project_root, input_path

    if raw_images_dir.exists():
        print("Cleaning existing raw_images...")
        shutil.rmtree(raw_images_dir)

    raw_images_dir.mkdir(parents=True, exist_ok=True)

    print("\nCopying images...")

    copied, skipped = 0, 0

    for img in input_path.iterdir():
        if not is_valid_image(img):
            skipped += 1
            continue

        try:
            shutil.copy2(img, raw_images_dir / img.name)
            copied += 1
        except Exception:
            skipped += 1

    print(f"\nCopied: {copied}, skipped: {skipped}")

    if copied == 0:
        raise RuntimeError("No valid images found")

    return project_root, raw_images_dir


# =====================================================
# PIPELINE SELECTION
# =====================================================
def get_pipeline_choice():
    print("\n=== Pipeline Selection ===")
    print("A → COLMAP (baseline)")
    print("B → GLOMAP (sparse) + COLMAP (dense)")

    choice = input("Select pipeline [A]: ").strip().upper()

    if choice not in ["A", "B"]:
        choice = "A"

    return choice


# =====================================================
# CONFIG BUILDER
# =====================================================
def get_user_config(project_root: Path, image_source: Path):

    print("\n=== Pipeline Configuration ===")

    retry_enabled = input("Enable adaptive retries? (y/n) [y]: ").strip().lower() != "n"

    max_retries = 2
    if retry_enabled:
        val = input("Max retries [2]: ").strip()
        if val:
            max_retries = int(val)

    user_config = {
        "paths": {
            "project_root": str(project_root)
        },

        "ingestion": {
            "external_image_path": str(image_source)
        },

        "downsampling": {
            "enabled": True
        },

        # 🔥 IMPORTANT: REMOVE AUTO BACKEND
        # pipeline decides backend now
        "sparse": {
            "backend": "colmap"  # default fallback
        },

        "retry": {
            "enabled": retry_enabled,
            "max_retries": max_retries
        },

        "system": {
            "ram_gb": 16,
            "gpu": True,
            "cpu_threads": -1
        }
    }

    return user_config


# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":

    project_root, image_source = get_user_paths()

    pipeline_choice = get_pipeline_choice()

    user_config = get_user_config(project_root, image_source)

    config = load_config(user_config)

    # 🔥 PASS PIPELINE TYPE
    runner = PipelineRunner(config, pipeline_type=pipeline_choice)

    runner.run()