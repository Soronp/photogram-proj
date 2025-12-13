import urllib.request
import zipfile
import os
import sys
import subprocess

# 1. Determine the current script's directory and move one folder up (project root)
current_dir = os.path.dirname(os.path.abspath(__file__))  # Get the folder where this script is located
project_root = os.path.dirname(current_dir)  # Move one folder up to the project root

# 2. Define the URL for COLMAP and the path where the ZIP file will be saved
url = "https://github.com/colmap/colmap/releases/download/3.12.6/colmap-x64-windows-cuda.zip"  # Updated URL for COLMAP 3.12.6
zip_path = os.path.join(project_root, "colmap.zip")  # Save the ZIP file in the project root folder
extract_path = os.path.join(project_root, "colmap")  # Extract the contents to the 'colmap' folder in the project root

# 3. Download COLMAP ZIP file
try:
    print(f"Downloading COLMAP from {url}...")
    urllib.request.urlretrieve(url, zip_path)
    print(f"Downloaded COLMAP to: {zip_path}")
except Exception as e:
    print(f"Error downloading COLMAP: {e}")
    sys.exit(1)

# 4. Extract the ZIP file
try:
    print(f"Extracting COLMAP to {extract_path}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)
    print(f"COLMAP extracted to: {extract_path}")
except Exception as e:
    print(f"Error extracting COLMAP: {e}")
    sys.exit(1)

# 5. Define the COLMAP bin path (the folder containing the executables)
colmap_bin_path = os.path.join(extract_path, "bin")

# 6. Add COLMAP bin folder to PATH for this session
os.environ["PATH"] += os.pathsep + colmap_bin_path

# 7. Inform the user that the process is complete
print(f"COLMAP bin folder added to PATH for this session: {colmap_bin_path}.")

# 8. Check if the folder was successfully added to PATH by running `colmap --version`
try:
    print("Checking COLMAP version...")
    result = subprocess.run(["colmap", "--version"], capture_output=True, text=True)
    if result.returncode == 0:
        print("Successfully added COLMAP to PATH. COLMAP version:")
        print(result.stdout)  # Print COLMAP version if successful
    else:
        print("Error: COLMAP is not in PATH.")
except FileNotFoundError:
    print("Error: COLMAP executable not found. Please check the PATH.")
    sys.exit(1)
