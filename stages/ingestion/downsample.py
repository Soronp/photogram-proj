from pathlib import Path
from PIL import Image
import shutil


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def resize_image(in_path, out_path, max_dim):
    with Image.open(in_path) as img:
        img = img.convert("RGB")

        w, h = img.size
        scale = min(max_dim / max(w, h), 1.0)

        if scale < 1.0:
            new_size = (int(w * scale), int(h * scale))
            img = img.resize(new_size, Image.LANCZOS)

        # 🔥 KEEP original format if possible
        img.save(out_path, quality=95)


def run(paths, config, logger):
    stage = "downsample"
    logger.info(f"---- {stage.upper()} ----")

    input_dir = paths.images
    temp_dir = paths.working / "_downsample_tmp"

    if not input_dir.exists():
        raise RuntimeError(f"{stage}: images not found")

    max_dim = config.get("downsampling", {}).get("target_max_dim", 2400)

    images = sorted([
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    ])

    if not images:
        raise RuntimeError(f"{stage}: no images found")

    logger.info(f"{stage}: processing {len(images)} images")

    # CLEAN TEMP
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)

    # -----------------------------
    # STEP 1: process
    # -----------------------------
    for img_path in images:
        out_path = temp_dir / img_path.name  # preserve name

        try:
            resize_image(img_path, out_path, max_dim)
        except Exception as e:
            raise RuntimeError(f"{stage}: failed {img_path.name} → {e}")

    # -----------------------------
    # STEP 2: SAFE REPLACEMENT
    # -----------------------------
    backup_dir = paths.working / "_backup_images"

    if backup_dir.exists():
        shutil.rmtree(backup_dir)

    shutil.move(str(input_dir), str(backup_dir))
    shutil.move(str(temp_dir), str(input_dir))

    shutil.rmtree(backup_dir)

    logger.info(f"{stage}: DONE (safe replace)")