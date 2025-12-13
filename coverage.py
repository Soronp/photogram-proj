import os
import shutil
from pathlib import Path
import cv2
import numpy as np
from utils.config import PATHS
from utils.logger import get_logger

logger = get_logger()


def compute_sift_features(image_path: Path):
    """
    Compute SIFT keypoints and descriptors for a single image.
    Returns (keypoints, descriptors) or (None, None) if failed.
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        logger.warning(f"Failed to read image: {image_path}")
        return None, None

    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)
    return keypoints, descriptors


def filter_redundant_images(input_folder: Path = None, output_folder: Path = None,
                            match_threshold: float = 0.7, min_images: int = 100):
    """
    Remove images that are too similar only if the dataset exceeds min_images.

    Args:
        input_folder: Path to normalized/processed images
        output_folder: Path to save filtered images
        match_threshold: fraction of matched features to consider image redundant
        min_images: only filter if number of images exceeds this

    Returns:
        Path to the folder containing filtered images.
    """
    # Use default paths if not provided
    if input_folder is None:
        input_folder = Path(PATHS['processed']) / "images_normalized"
    if output_folder is None:
        output_folder = Path(PATHS['filtered'])
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    images = sorted([f for f in input_folder.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")])
    total_images = len(images)

    if total_images == 0:
        logger.warning(f"No images found in {input_folder}")
        return output_folder

    # Only apply filter if more than min_images
    if total_images <= min_images:
        logger.info(f"Image count ({total_images}) <= {min_images}, skipping coverage filter.")
        for img_path in images:
            shutil.copy2(img_path, output_folder / img_path.name)
        return output_folder

    logger.info(f"Computing SIFT descriptors for {total_images} images...")
    descriptors_list = []
    image_paths = []

    # Step 1: Compute descriptors
    for img_path in images:
        kp, des = compute_sift_features(img_path)
        if des is not None:
            descriptors_list.append(des)
            image_paths.append(img_path)

    # Step 2: Compare each image to previous ones
    bf = cv2.BFMatcher()
    keep_images = []
    for i, des1 in enumerate(descriptors_list):
        redundant = False
        for j in range(i):
            des2 = descriptors_list[j]
            if des2 is None:
                continue
            matches = bf.knnMatch(des1, des2, k=2)
            # Ratio test as per Lowe
            good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
            if len(good_matches) / len(des1) > match_threshold:
                redundant = True
                logger.info(f"Image {image_paths[i].name} is redundant with {image_paths[j].name}")
                break
        if not redundant:
            keep_images.append(image_paths[i])

    # Step 3: Copy non-redundant images to output folder
    for img_path in keep_images:
        shutil.copy2(img_path, output_folder / img_path.name)

    logger.info(f"Filtered images saved to {output_folder}")
    logger.info(f"Kept {len(keep_images)} out of {total_images} images")
    return output_folder


def run_coverage_filter():
    """
    Entry point for orchestrator.
    Returns path to filtered images folder.
    """
    filtered_folder = filter_redundant_images()
    # Return string for JSON checkpoint safety
    return str(filtered_folder)


if __name__ == "__main__":
    run_coverage_filter()
