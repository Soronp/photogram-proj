from pathlib import Path
from core.runner import PipelineRunner
from config.config_manager import load_config


def get_user_paths():
    print("=== Photogrammetry Pipeline Setup ===")

    raw_input_path = input("Enter path to your image folder: ").strip()

    if not raw_input_path:
        raise ValueError("Input path cannot be empty")

    raw_path = Path(raw_input_path)

    if not raw_path.exists():
        raise FileNotFoundError(f"Path does not exist: {raw_path}")

    project_root = Path("data/projects/project_1")

    # Create required structure
    raw_images_dir = project_root / "raw_images"
    raw_images_dir.mkdir(parents=True, exist_ok=True)

    print("Copying images into project structure...")

    for img in raw_path.iterdir():
        target = raw_images_dir / img.name
        if not target.exists():
            try:
                target.write_bytes(img.read_bytes())
            except Exception:
                print(f"Skipping invalid file: {img.name}")

    print(f"Images prepared at: {raw_images_dir}")

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
        }
    }

    config = load_config(user_config)

    runner = PipelineRunner(config)
    runner.run()