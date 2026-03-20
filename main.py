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

    # -----------------------------
    # INPUT
    # -----------------------------
    input_path = Path(input("Enter path to your IMAGE folder: ").strip())

    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    # -----------------------------
    # OUTPUT
    # -----------------------------
    project_root = Path(input("Enter path for OUTPUT project folder: ").strip())

    raw_images_dir = project_root / "raw_images"

    # -----------------------------
    # COPY MODE SELECTION
    # -----------------------------
    copy_choice = input("Copy images into project? (y/n) [y]: ").strip().lower()
    copy_enabled = (copy_choice != "n")

    if not copy_enabled:
        print("Using images in-place (no copying)")
        return project_root, input_path

    # -----------------------------
    # CLEAN TARGET FOLDER (CRITICAL FIX)
    # -----------------------------
    if raw_images_dir.exists():
        print("Cleaning existing raw_images folder...")
        shutil.rmtree(raw_images_dir)

    raw_images_dir.mkdir(parents=True, exist_ok=True)

    print("\nCopying images...")

    copied, skipped = 0, 0

    for img in input_path.iterdir():

        if not is_valid_image(img):
            skipped += 1
            continue

        target = raw_images_dir / img.name

        try:
            shutil.copy2(img, target)
            copied += 1
        except Exception:
            print(f"Skipping invalid file: {img.name}")
            skipped += 1

    print(f"\nImages copied: {copied}, skipped: {skipped}")

    if copied == 0:
        raise RuntimeError("No valid images were copied. Check file formats.")

    print(f"Project created at: {project_root}\n")

    return project_root, raw_images_dir


# =====================================================
# CONFIG BUILDER
# =====================================================
def get_user_config(project_root: Path, image_source: Path):

    print("=== Pipeline Configuration ===")

    # -----------------------------
    # RETRY SYSTEM
    # -----------------------------
    retry_enabled = input("Enable retry system? (y/n) [y]: ").strip().lower() != "n"

    max_retries = 2
    if retry_enabled:
        val = input("Max retries [2]: ").strip()
        if val:
            max_retries = int(val)

    # -----------------------------
    # DOWNSAMPLING
    # -----------------------------
    downsample_enabled = input("Enable downsampling? (y/n) [y]: ").strip().lower() != "n"

    target_dim = 2000
    if downsample_enabled:
        val = input("Target max dimension [2000]: ").strip()
        if val:
            target_dim = int(val)

    # -----------------------------
    # BACKEND
    # -----------------------------
    backend = input("Backend (colmap/glomap) [colmap]: ").strip().lower()
    if backend not in ["colmap", "glomap"]:
        backend = "colmap"

    # -----------------------------
    # CONFIG
    # -----------------------------
    user_config = {
        "paths": {
            "project_root": str(project_root)
        },

        "ingestion": {
            # 🔥 critical: allow external image source
            "external_image_path": str(image_source)
        },

        "downsampling": {
            "enabled": downsample_enabled,
            "target_max_dim": target_dim
        },

        "sparse": {
            "backend": backend
        },

        "retry": {
            "enabled": retry_enabled,
            "max_retries": max_retries
        }
    }

    return user_config


# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":

    project_root, image_source = get_user_paths()

    user_config = get_user_config(project_root, image_source)

    config = load_config(user_config)

    runner = PipelineRunner(config)
    runner.run()