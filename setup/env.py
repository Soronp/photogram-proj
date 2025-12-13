import subprocess
import sys
import os

# 1. Get the current working directory and navigate to the parent folder (project folder)
current_dir = os.path.dirname(os.path.abspath(__file__))  # This gets the folder where the script is located
project_folder = os.path.dirname(current_dir)  # The parent folder (project folder)

# 2. Define the path to the setup folder and the env.py script
setup_folder = os.path.join(project_folder, "setup")
env_script_path = os.path.join(setup_folder, "env.py")

# Ensure the env.py script exists at the path
if not os.path.exists(env_script_path):
    print(f"Error: {env_script_path} does not exist.")
    sys.exit(1)

# 3. Create virtual environment one folder before (in the parent directory)
venv_path = os.path.join(project_folder, "venv")  # Create the virtual environment in the project folder
subprocess.run([sys.executable, "-m", "venv", venv_path])

# 4. Define pip executable
if os.name == "nt":  # Windows
    pip_exe = os.path.join(venv_path, "Scripts", "pip.exe")
else:  # Linux/Mac
    pip_exe = os.path.join(venv_path, "bin", "pip")

# 5. Upgrade pip
subprocess.run([pip_exe, "install", "--upgrade", "pip"])

# 6. Install core packages
packages = [
    "numpy", "opencv-python", "pillow", "open3d",
    "matplotlib", "pandas", "shapely", "tqdm"
]
subprocess.run([pip_exe, "install"] + packages)

# 7. Notify that the virtual environment is ready
print("Python virtual environment ready with core packages.")

# 8. Run env.py to perform any additional setup
subprocess.run([sys.executable, env_script_path])

print(f"Environment setup completed. You can now run {env_script_path} if necessary.")
