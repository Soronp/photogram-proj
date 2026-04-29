from pathlib import Path
import shutil
import json

from core.runner import PipelineRunner
from config.config_manager import load_config


# =====================================================
# CONSTANTS
# =====================================================
VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def is_valid_image(file: Path):
    return file.is_file() and file.suffix.lower() in VALID_EXT


# =====================================================
# FIND EXISTING RUNS
# =====================================================
def find_existing_runs(project_root: Path):
    runs = []
    if not project_root.exists():
        return runs

    for p in project_root.iterdir():
        if not p.is_dir() or not p.name.startswith("run_"):
            continue

        if (p / "pipeline_state.json").exists():
            runs.append(p)

        nested_root = p / "runs"
        if nested_root.exists():
            for sub in nested_root.iterdir():
                if sub.is_dir() and (sub / "pipeline_state.json").exists():
                    runs.append(sub)

    return sorted(runs)


# =====================================================
# SELECT OR CREATE RUN
# =====================================================
def select_or_create_run(project_root: Path):
    runs = find_existing_runs(project_root)

    if runs:
        print("\n⚠️ Existing runs found:\n")

        for i, run in enumerate(runs):
            print(f"[{i}] {run}")

        print("\nOptions:")
        print("R → Resume existing run")
        print("N → Start new run")

        choice = input("Select option [R/N]: ").strip().upper()

        if choice == "R":
            idx = int(input("Enter run index: ").strip())

            if idx < 0 or idx >= len(runs):
                raise ValueError("Invalid run index")

            selected_run = runs[idx]
            print(f"✔ Resuming {selected_run}")

            return selected_run, True

    project_root.mkdir(parents=True, exist_ok=True)

    existing_ids = [
        int(p.name.split("_")[1])
        for p in runs if p.name.split("_")[1].isdigit()
    ] if runs else []

    next_id = max(existing_ids, default=0) + 1
    new_run = project_root / f"run_{next_id:03d}"
    new_run.mkdir(parents=True, exist_ok=True)

    print(f"✔ Created new run: {new_run.name}")

    return new_run, False


# =====================================================
# INPUT HANDLING
# =====================================================
def get_user_paths():
    print("=== Photogrammetry Pipeline Setup ===")

    input_path = Path(input("Enter path to IMAGE folder: ").strip())
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    project_root = Path(input("Enter PROJECT root folder: ").strip())

    run_path, is_resume = select_or_create_run(project_root)
    raw_images_dir = run_path / "raw_images"

    if is_resume:
        print("✔ Using existing images")
        return run_path, raw_images_dir

    raw_images_dir.mkdir(parents=True, exist_ok=True)

    copy_choice = input("Copy images into run folder? (y/n) [y]: ").strip().lower()
    copy_enabled = (copy_choice != "n")

    if not copy_enabled:
        return run_path, input_path

    print("\nCopying images...")
    count = 0

    for img in input_path.iterdir():
        if is_valid_image(img):
            shutil.copy2(img, raw_images_dir / img.name)
            count += 1

    print(f"Copied {count} images")

    return run_path, raw_images_dir


# =====================================================
# PIPELINE SELECTION (E REMOVED)
# =====================================================
def get_pipeline_choice():
    print("\n=== Pipeline Selection ===")
    print("A → COLMAP full")
    print("B → GLOMAP + COLMAP dense")
    print("C → COLMAP + OpenMVS")
    print("D → OpenMVG (SfM only)")

    choice = input("Select pipeline [A]: ").strip().upper()
    return choice if choice in ["A", "B", "C", "D"] else "A"


# =====================================================
# CONFIG BUILDER (FIXED)
# =====================================================
def get_user_config(run_path: Path, image_source: Path, pipeline_choice: str):

    user_config = {
        "paths": {"project_root": str(run_path)},
        "ingestion": {"external_image_path": str(image_source)},
        "downsampling": {"enabled": True},
    }

    # ----------------------------
    # C → COLMAP + OpenMVS
    # ----------------------------
    if pipeline_choice == "C":
        user_config.setdefault("pipeline", {})
        user_config["pipeline"]["backends"] = {
            "sparse": "colmap",
            "dense": "openmvs",
            "mesh": "openmvs",
            "texture": "openmvs"
        }

    # ----------------------------
    # D → OPENMVG SfM ONLY (FIXED)
    # ----------------------------
    elif pipeline_choice == "D":
        user_config.setdefault("pipeline", {})
        user_config["pipeline"]["backends"] = {
            "sparse": "openmvg",
            "dense": "disabled",
            "mesh": "disabled",
            "texture": "disabled"
        }

    return user_config


# =====================================================
# MAIN ENTRY
# =====================================================
if __name__ == "__main__":
    try:
        run_path, image_source = get_user_paths()
        pipeline_choice = get_pipeline_choice()

        user_config = get_user_config(run_path, image_source, pipeline_choice)
        config = load_config(user_config)

        runner = PipelineRunner(
            config,
            run_root=run_path,
            pipeline_type=pipeline_choice
        )

        runner.run()

        print("\n✅ Pipeline completed successfully!")

    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")