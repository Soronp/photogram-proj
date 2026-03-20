from pathlib import Path
import shutil


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def run(paths, config, logger):
    stage = "ingest_images"
    logger.info(f"---- {stage.upper()} ----")

    raw_dir = paths.raw_images
    working_dir = paths.images

    # 🔥 SUPPORT EXTERNAL IMAGE SOURCE
    external_path = config.get("ingestion", {}).get("external_image_path")

    if external_path:
        raw_dir = Path(external_path)
        logger.info(f"{stage}: using external image path → {raw_dir}")

    if not raw_dir.exists():
        raise RuntimeError(f"{stage}: raw_images folder not found")

    # 🔥 CLEAN working directory (CRITICAL)
    if working_dir.exists():
        shutil.rmtree(working_dir)

    working_dir.mkdir(parents=True, exist_ok=True)

    # 🔥 Normalize extensions
    images = sorted([
        p for p in raw_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    ])

    if not images:
        raise RuntimeError(f"{stage}: no valid images found")

    logger.info(f"{stage}: found {len(images)} images")

    copy_mode = config.get("ingestion", {}).get("copy_mode", "copy")

    copied = 0

    for img_path in images:
        target_path = working_dir / img_path.name

        try:
            if copy_mode == "copy":
                shutil.copy2(img_path, target_path)

            elif copy_mode == "symlink":
                try:
                    target_path.symlink_to(img_path.resolve())
                except Exception:
                    shutil.copy2(img_path, target_path)

            else:
                raise ValueError(f"{stage}: unknown copy_mode")

            copied += 1

        except Exception as e:
            logger.warning(f"{stage}: failed {img_path.name} → {e}")

    if copied == 0:
        raise RuntimeError(f"{stage}: no images copied")

    logger.info(f"{stage}: copied={copied}")
    logger.info(f"{stage}: DONE")