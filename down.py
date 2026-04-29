#!/usr/bin/env python3
from pathlib import Path
from PIL import Image
import sys

MAX_SIZE = 2000  # max width/height constraint


def downsample_image(img: Image.Image, max_size: int):
    """
    Resize image while preserving aspect ratio.
    """
    w, h = img.size

    if max(w, h) <= max_size:
        return img  # no resize needed

    scale = max_size / max(w, h)
    new_size = (int(w * scale), int(h * scale))

    return img.resize(new_size, Image.LANCZOS)


def process_image(in_path: Path, out_path: Path):
    try:
        with Image.open(in_path) as img:

            # Preserve EXIF if available (JPEG only typically)
            exif = img.info.get("exif")

            # Convert mode safety (prevents save issues)
            if img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGB")

            img_resized = downsample_image(img, MAX_SIZE)

            out_path.parent.mkdir(parents=True, exist_ok=True)

            save_kwargs = {}

            # Preserve EXIF only for formats that support it (JPEG)
            if exif and in_path.suffix.lower() in [".jpg", ".jpeg"]:
                save_kwargs["exif"] = exif

            # Keep quality stable for photogrammetry pipelines
            if in_path.suffix.lower() in [".jpg", ".jpeg"]:
                save_kwargs["quality"] = 95
                save_kwargs["optimize"] = True

            img_resized.save(out_path, **save_kwargs)

            print(f"[OK] {in_path.name} -> {out_path.name}")

    except Exception as e:
        print(f"[FAIL] {in_path.name}: {e}")


def main():
    print("\n=== Image Downsampler (MAX 2K, EXIF-safe) ===")

    input_dir = Path(input("Input folder: ").strip())
    output_dir = Path(input("Output folder: ").strip())

    if not input_dir.exists():
        print("[ERROR] Input folder does not exist")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    valid_ext = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

    images = [p for p in input_dir.iterdir()
              if p.is_file() and p.suffix.lower() in valid_ext]

    if not images:
        print("[ERROR] No valid images found")
        sys.exit(1)

    print(f"[INFO] Found {len(images)} images")
    print(f"[INFO] Max size = {MAX_SIZE}px")

    for img_path in images:
        out_path = output_dir / img_path.name
        process_image(img_path, out_path)

    print("\n[DONE] Downsampling complete")


if __name__ == "__main__":
    main()