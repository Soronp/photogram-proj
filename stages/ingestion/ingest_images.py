from pathlib import Path
import shutil


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def run(paths, config, logger):
    stage = "ingest_images"
    logger.info(f"---- {stage.upper()} ----")

    raw_dir = paths.raw_images
    working_dir = paths.images

    if not raw_dir.exists():
        raise RuntimeError(f"{stage}: raw_images folder not found at {raw_dir}")

    working_dir.mkdir(parents=True, exist_ok=True)

    copy_mode = config.get("ingestion", {}).get("copy_mode", "copy")

    images = [p for p in raw_dir.iterdir() if p.suffix in VALID_EXTENSIONS]

    if not images:
        raise RuntimeError(f"{stage}: No valid images found in {raw_dir}")

    logger.info(f"{stage}: Found {len(images)} images")

    copied = 0
    skipped = 0

    for img_path in images:
        target_path = working_dir / img_path.name

        if target_path.exists():
            skipped += 1
            continue

        if copy_mode == "copy":
            shutil.copy2(img_path, target_path)

        elif copy_mode == "symlink":
            try:
                target_path.symlink_to(img_path.resolve())
            except Exception:
                logger.warning(f"{stage}: symlink failed, falling back to copy")
                shutil.copy2(img_path, target_path)

        else:
            raise ValueError(f"{stage}: Unknown copy_mode: {copy_mode}")

        copied += 1

    logger.info(f"{stage}: copied={copied}, skipped={skipped}")
    logger.info(f"{stage}: DONE")