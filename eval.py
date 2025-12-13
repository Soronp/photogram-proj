import os
import numpy as np
import open3d as o3d
import pandas as pd
import matplotlib.pyplot as plt
from utils.config import PATHS
from utils.logger import get_logger
import json

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


# -------------------------------
# Evaluation functions
# -------------------------------
def evaluate_pointcloud(ply_path):
    """Compute basic metrics for a point cloud."""
    ply_path = str(ply_path)
    if not os.path.exists(ply_path):
        logger.warning(f"Point cloud not found: {ply_path}")
        return None
    pcd = o3d.io.read_point_cloud(ply_path)
    num_points = np.asarray(pcd.points).shape[0]
    bbox = pcd.get_axis_aligned_bounding_box()
    volume = bbox.volume()
    return {"num_points": num_points, "bounding_box_volume": volume}


def evaluate_mesh(mesh_path):
    """Compute basic metrics for a mesh."""
    mesh_path = str(mesh_path)
    if not os.path.exists(mesh_path):
        logger.warning(f"Mesh not found: {mesh_path}")
        return None
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    num_vertices = len(mesh.vertices)
    num_triangles = len(mesh.triangles)
    bbox = mesh.get_axis_aligned_bounding_box()
    volume = bbox.volume()
    return {"num_vertices": num_vertices, "num_triangles": num_triangles, "bounding_box_volume": volume}


def generate_visual_heatmap(ply_path, output_path):
    """Visualize point cloud density as a 2D heatmap (XY plane)."""
    ply_path = str(ply_path)
    if not os.path.exists(ply_path):
        logger.warning(f"Point cloud not found for heatmap: {ply_path}")
        return
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)
    x = points[:, 0]
    y = points[:, 1]

    plt.figure(figsize=(6, 6))
    plt.hexbin(x, y, gridsize=100, cmap='inferno')
    plt.colorbar(label='Point Density')
    plt.title('Point Cloud Density Heatmap')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Heatmap saved to: {output_path}")


# -------------------------------
# Orchestrator
# -------------------------------
def run_evaluation():
    os.makedirs(PATHS['evaluations'], exist_ok=True)
    metrics = {}
    checkpoint = load_checkpoint()

    # Sparse point cloud
    sparse_ply = checkpoint.get("sparse") or os.path.join(PATHS['sparse'], "sparse_model.ply")
    sparse_metrics = evaluate_pointcloud(sparse_ply)
    if sparse_metrics:
        metrics['sparse'] = sparse_metrics
    else:
        logger.warning("Sparse point cloud metrics skipped.")

    # Dense point cloud
    dense_ply = checkpoint.get("dense") or os.path.join(PATHS['dense'], "fused.ply")
    dense_metrics = evaluate_pointcloud(dense_ply)
    if dense_metrics:
        metrics['dense'] = dense_metrics
        heatmap_path = os.path.join(PATHS['evaluations'], "dense_heatmap.png")
        generate_visual_heatmap(dense_ply, heatmap_path)
    else:
        logger.warning("Dense point cloud metrics skipped.")

    # Mesh
    mesh_ply = checkpoint.get("mesh") or os.path.join(PATHS['mesh'], "mesh.ply")
    mesh_metrics = evaluate_mesh(mesh_ply)
    if mesh_metrics:
        metrics['mesh'] = mesh_metrics
    else:
        logger.warning("Mesh metrics skipped.")

    # Save metrics to CSV
    df_list = []
    for key, val in metrics.items():
        row = val.copy()
        row['type'] = key
        df_list.append(row)
    df = pd.DataFrame(df_list)
    csv_path = os.path.join(PATHS['evaluations'], "evaluation_metrics.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"Evaluation metrics saved to: {csv_path}")

    return metrics


if __name__ == "__main__":
    run_evaluation()
