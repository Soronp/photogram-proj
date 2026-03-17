from pathlib import Path
from PIL import Image, ImageOps

from core.stage import Stage


class ImageDownsampleStage(Stage):

    name = "image_downsample"

    def __init__(self, max_size: int = 2048):
        self.max_size = max_size

    def run(self, paths, config, logger, tool_runner):

        input_dir = Path(paths.images)
        output_dir = Path(paths.preprocessed_images)

        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Starting image downsample stage")

        supported = [".jpg", ".jpeg", ".png"]

        images = sorted(
            p for p in input_dir.iterdir()
            if p.suffix.lower() in supported
        )

        if not images:
            raise RuntimeError("No images found")

        processed = 0

        for img_path in images:

            out_path = output_dir / img_path.name

            try:

                img = Image.open(img_path)

                exif = img.info.get("exif")
                icc = img.info.get("icc_profile")

                img = ImageOps.exif_transpose(img)

                w, h = img.size
                max_dim = max(w, h)

                if max_dim > self.max_size:

                    scale = self.max_size / max_dim

                    new_size = (
                        int(w * scale),
                        int(h * scale)
                    )

                    img = img.resize(new_size, Image.LANCZOS)

                    logger.info(
                        f"{img_path.name}: {w}x{h} -> {new_size}"
                    )

                save_args = dict(
                    quality=95,
                    subsampling=0
                )

                if exif:
                    save_args["exif"] = exif

                if icc:
                    save_args["icc_profile"] = icc

                img.save(out_path, **save_args)

                processed += 1

            except Exception as e:
                logger.warning(f"{img_path.name} failed: {e}")

        logger.info(f"Processed {processed} images")

        paths.images = output_dir