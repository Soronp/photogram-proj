import os

# Define folder structure
folders = [
    "input/images",
    "input/videos",
    "processed/images_normalized",
    "processed/images_filtered",
    "reconstructions/sparse",
    "reconstructions/dense",
    "reconstructions/mesh",
    "logs",
    "evaluations",
    "utils"
]
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))

for f in folders:
    path = os.path.join(project_root, f)
    os.makedirs(path, exist_ok=True)
    print(f"Created: {path}")

print("Project structure created successfully.")
