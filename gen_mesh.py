import os
import json
import numpy as np
import open3d as o3d
from utils.config import PATHS
from utils.logger import get_logger

logger = get_logger()
CHECKPOINT_FILE = os.path.join(PATHS['logs'], "pipeline_checkpoint.json")

# -------------------------------
# Checkpoint helpers
# -------------------------------

def load_checkpoint():
    """Load pipeline checkpoint if exists."""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            return json.load(f)
    return {}

def save_checkpoint(step_name, output):
    """Save current step to checkpoint."""
    checkpoint = load_checkpoint()
    checkpoint[step_name] = str(output)
    os.makedirs(PATHS['logs'], exist_ok=True)
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint, f, indent=2)
    logger.info(f"Checkpoint saved: {step_name}")

# -------------------------------
# Mesh generation
# -------------------------------

def generate_mesh_from_pointcloud(
    ply_path,
    output_folder=None,
    poisson_depth=10,
    simplify_target_triangles=50000,
    density_crop_quantile=0.01
):
    """Generate mesh from dense point cloud with Poisson reconstruction."""
    ply_path = str(ply_path)
    if not os.path.exists(ply_path):
        logger.error(f"Point cloud file not found: {ply_path}")
        return None

    if output_folder is None:
        output_folder = PATHS['mesh']
    os.makedirs(output_folder, exist_ok=True)

    logger.info(f"Loading point cloud from: {ply_path}")
    pcd = o3d.io.read_point_cloud(ply_path)
    if not pcd.has_normals():
        logger.info("Estimating normals...")
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))

    logger.info("Running Poisson surface reconstruction...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=poisson_depth)

    densities = np.asarray(densities)
    vertices_to_keep = densities > np.quantile(densities, density_crop_quantile)
    mesh = mesh.select_by_index(np.where(vertices_to_keep)[0])
    logger.info(f"Removed low-density vertices below quantile {density_crop_quantile}")

    logger.info(f"Simplifying mesh to ~{simplify_target_triangles} triangles...")
    mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=simplify_target_triangles)

    output_path = os.path.join(output_folder, "mesh.ply")
    o3d.io.write_triangle_mesh(output_path, mesh)
    logger.info(f"Mesh generated and saved to: {output_path}")
    return output_path

# -------------------------------
# Orchestrator entry
# -------------------------------

def run_mesh_generation(dense_ply_path=None):
    """Generate mesh with checkpointing, using dense reconstruction output if needed."""
    checkpoint = load_checkpoint()
    if "mesh" in checkpoint:
        logger.info("Mesh already generated. Skipping.")
        return checkpoint["mesh"]

    # Use dense reconstruction checkpoint if available
    if dense_ply_path is None:
        dense_ply_path = checkpoint.get("dense")
        if dense_ply_path is None:
            # fallback path
            dense_ply_path = os.path.join(PATHS['dense'], "fused.ply")

    dense_ply_path = str(dense_ply_path)
    if not os.path.exists(dense_ply_path):
        logger.error(f"Dense point cloud not found: {dense_ply_path}")
        return None

    mesh_path = generate_mesh_from_pointcloud(dense_ply_path)
    if mesh_path:
        save_checkpoint("mesh", mesh_path)
    return mesh_path

# -------------------------------
# Standalone execution
# -------------------------------

if __name__ == "__main__":
    run_mesh_generation()
