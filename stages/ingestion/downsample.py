from pathlib import Path
from PIL import Image


def resize_image(in_path, out_path, max_dim):
    with Image.open(in_path) as img:
        w, h = img.size

        scale = min(max_dim / max(w, h), 1.0)

        new_w = int(w * scale)
        new_h = int(h * scale)

        if scale < 1.0:
            img = img.resize((new_w, new_h), Image.LANCZOS)

        img.save(out_path, quality=95)


def run(paths, config, logger):
    stage = "downsample"
    logger.info(f"---- {stage.upper()} ----")

    input_dir = paths.images
    output_dir = paths.images_downsampled

    if not input_dir.exists():
        raise RuntimeError(f"{stage}: input images not found")

    output_dir.mkdir(parents=True, exist_ok=True)

    ds_config = config.get("downsampling", {})
    max_dim = ds_config.get("target_max_dim", 2000)

    images = list(input_dir.iterdir())

    processed = 0
    skipped = 0

    for img_path in images:
        out_path = output_dir / img_path.name

        if out_path.exists():
            skipped += 1
            continue

        try:
            resize_image(img_path, out_path, max_dim)
            processed += 1
        except Exception as e:
            logger.warning(f"{stage}: failed {img_path.name} -> {e}")

    logger.info(f"{stage}: processed={processed}, skipped={skipped}")
    logger.info(f"{stage}: DONE")