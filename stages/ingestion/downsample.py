from pathlib import Path
from PIL import Image
import shutil


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def resize_image(in_path, out_path, max_dim):
    with Image.open(in_path) as img:
        img = img.convert("RGB")  # ensure consistent format

        w, h = img.size
        scale = min(max_dim / max(w, h), 1.0)

        new_w = int(w * scale)
        new_h = int(h * scale)

        if scale < 1.0:
            img = img.resize((new_w, new_h), Image.LANCZOS)

        img.save(out_path, format="JPEG", quality=95)


def run(paths, config, logger):
    stage = "downsample"
    logger.info(f"---- {stage.upper()} ----")

    input_dir = paths.images
    temp_dir = paths.working / "_downsample_tmp"

    if not input_dir.exists():
        raise RuntimeError(f"{stage}: input images not found")

    temp_dir.mkdir(parents=True, exist_ok=True)

    max_dim = config.get("downsampling", {}).get("target_max_dim", 2000)

    # 🔒 deterministic order
    images = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in VALID_EXTENSIONS])

    if not images:
        raise RuntimeError(f"{stage}: no images found")

    logger.info(f"{stage}: processing {len(images)} images")

    processed = 0

    # -----------------------------
    # STEP 1: downsample into temp
    # -----------------------------
    for img_path in images:
        out_path = temp_dir / (img_path.stem + ".jpg")  # normalize to jpg

        try:
            resize_image(img_path, out_path, max_dim)
            processed += 1
        except Exception as e:
            raise RuntimeError(f"{stage}: failed on {img_path.name} -> {e}")

    if processed != len(images):
        raise RuntimeError(f"{stage}: mismatch in processed images")

    # -----------------------------
    # STEP 2: replace originals
    # -----------------------------
    logger.info(f"{stage}: replacing original images")

    for img in input_dir.iterdir():
        img.unlink()

    for img in sorted(temp_dir.iterdir()):
        shutil.move(str(img), str(input_dir / img.name))

    temp_dir.rmdir()

    logger.info(f"{stage}: DONE (all images now standardized)")