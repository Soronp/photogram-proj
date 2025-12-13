import os
import cv2
import hashlib
from utils.config import PATHS
from utils.logger import get_logger
import subprocess

logger = get_logger()

def get_image_files(folder):
    """Return list of image file paths in folder"""
    exts = (".jpg", ".jpeg", ".png", ".tif")
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(exts)]

def normalize_image(img_path, output_folder, resize=(1024, 768)):
    """Read, resize, and save image to output folder"""
    os.makedirs(output_folder, exist_ok=True)
    img = cv2.imread(img_path)
    if img is None:
        logger.warning(f"Failed to read: {img_path}")
        return None
    img_resized = cv2.resize(img, resize)
    filename = os.path.basename(img_path)
    save_path = os.path.join(output_folder, filename)
    cv2.imwrite(save_path, img_resized)
    logger.info(f"Normalized image: {filename}")
    return save_path

def remove_duplicates(image_paths):
    """Remove exact duplicates using hashing"""
    hash_dict = {}
    unique_images = []
    for path in image_paths:
        with open(path, "rb") as f:
            h = hashlib.md5(f.read()).hexdigest()
        if h not in hash_dict:
            hash_dict[h] = path
            unique_images.append(path)
        else:
            logger.info(f"Duplicate removed: {os.path.basename(path)}")
            os.remove(path)
    return unique_images

def extract_frames_from_videos(video_folder, output_folder, frame_rate=1, use_ffmpeg=False):
    """Extract frames from videos using OpenCV or ffmpeg"""
    os.makedirs(output_folder, exist_ok=True)
    videos = [f for f in os.listdir(video_folder) if f.lower().endswith((".mp4", ".mov", ".avi"))]
    
    for vid in videos:
        video_path = os.path.join(video_folder, vid)
        if use_ffmpeg:
            # Using ffmpeg (faster and more reliable for large videos)
            output_pattern = os.path.join(output_folder, f"{os.path.splitext(vid)[0]}_frame%05d.jpg")
            cmd = [
                "ffmpeg",
                "-i", video_path,
                "-vf", f"fps={frame_rate}",
                output_pattern
            ]
            logger.info(f"Running ffmpeg for {vid}")
            subprocess.run(cmd, check=True)
        else:
            # Fallback: OpenCV extraction
            cap = cv2.VideoCapture(video_path)
            frame_count = 0
            saved_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_count % frame_rate == 0:
                    filename = f"{os.path.splitext(vid)[0]}_frame{frame_count:05d}.jpg"
                    save_path = os.path.join(output_folder, filename)
                    cv2.imwrite(save_path, frame)
                    saved_count += 1
                frame_count += 1
            cap.release()
            logger.info(f"Extracted {saved_count} frames from {vid}")

def run_input_handler(frame_rate=1, use_ffmpeg=False):
    """Main function for automated input handling"""
    logger.info("Starting input handling...")

    # Step 1: Extract frames from videos (if any)
    if os.path.exists(PATHS['input_videos']):
        extract_frames_from_videos(PATHS['input_videos'], PATHS['input_images'], frame_rate, use_ffmpeg)

    # Step 2: Normalize images
    image_files = get_image_files(PATHS['input_images'])
    for img_path in image_files:
        normalize_image(img_path, PATHS['normalized'])

    # Step 3: Remove duplicates
    normalized_files = get_image_files(PATHS['normalized'])
    remove_duplicates(normalized_files)

    logger.info("Input handling finished successfully.")

if __name__ == "__main__":
    run_input_handler(frame_rate=5, use_ffmpeg=True)
