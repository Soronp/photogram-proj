from pathlib import Path
import shutil

from core.runner import PipelineRunner
from config.config_manager import load_config


def get_user_paths():
    print("=== Photogrammetry Pipeline Setup ===")

    # -----------------------------
    # INPUT DATASET
    # -----------------------------
    input_path_str = input("Enter path to your IMAGE folder: ").strip()

    if not input_path_str:
        raise ValueError("Input path cannot be empty")

    input_path = Path(input_path_str)

    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    # -----------------------------
    # OUTPUT PROJECT ROOT
    # -----------------------------
    output_path_str = input("Enter path for OUTPUT project folder: ").strip()

    if not output_path_str:
        raise ValueError("Output path cannot be empty")

    project_root = Path(output_path_str)

    # Create project structure
    raw_images_dir = project_root / "raw_images"
    raw_images_dir.mkdir(parents=True, exist_ok=True)

    print("\nPreparing project structure...")

    # -----------------------------
    # COPY IMAGES (SAFE METHOD)
    # -----------------------------
    copied = 0
    skipped = 0

    for img in input_path.iterdir():
        target = raw_images_dir / img.name

        if target.exists():
            skipped += 1
            continue

        try:
            shutil.copy2(img, target)
            copied += 1
        except Exception:
            print(f"Skipping invalid file: {img.name}")

    print(f"\nImages copied: {copied}, skipped: {skipped}")
    print(f"Project created at: {project_root}\n")

    return project_root


if __name__ == "__main__":
    project_root = get_user_paths()

    user_config = {
        "paths": {
            "project_root": str(project_root)
        },
        "downsampling": {
            "enabled": True,
            "target_max_dim": 2000
        },
        "sparse": {
            "backend": "colmap"   # or "glomap"
        }
    }

    config = load_config(user_config)

    runner = PipelineRunner(config)
    runner.run()