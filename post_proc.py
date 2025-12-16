#!/usr/bin/env python3
"""
mesh_postprocess.py

MARK-2 Mesh Post-Processing Stage
--------------------------------
- Cleans and stabilizes raw surface mesh
- Removes degenerate geometry and small components
- Optional smoothing and decimation (configurable, deterministic)
- Input : mesh/mesh_raw.ply
- Output: mesh/mesh_cleaned.ply
- Fully logged, restart-safe, force-compatible
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import open3d as o3d

from utils.logger import get_logger
from utils.paths import ProjectPaths
from utils.config import load_config


# ------------------------------------------------------------------
# Mesh post-processing pipeline
# ------------------------------------------------------------------
def run_mesh_postprocess(project_root: Path, force: bool):
    paths = ProjectPaths(project_root)
    config = load_config(project_root)
    logger = get_logger("mesh_postprocess", project_root)

    mesh_dir = paths.mesh
    mesh_in: Path = mesh_dir / "mesh_raw.ply"
    mesh_out: Path = mesh_dir / "mesh_cleaned.ply"
    report_out: Path = mesh_dir / "mesh_postprocess_report.json"

    logger.info("=== Mesh Post-Processing Stage ===")
    logger.info(f"Mesh input  : {mesh_in}")
    logger.info(f"Mesh output : {mesh_out}")

    if not mesh_in.exists():
        raise FileNotFoundError(f"Raw mesh not found: {mesh_in}")

    if mesh_out.exists() and not force:
        logger.info("Cleaned mesh already exists — skipping (use --force to regenerate)")
        return

    # --------------------------------------------------
    # Load raw mesh
    # --------------------------------------------------
    logger.info("Loading raw mesh")
    mesh = o3d.io.read_triangle_mesh(str(mesh_in))
    if not mesh.has_triangles():
        raise RuntimeError("Input mesh contains no triangles")

    mesh.compute_vertex_normals()

    report = {
        "input_vertices": len(mesh.vertices),
        "input_triangles": len(mesh.triangles),
    }
    logger.info(f"Loaded mesh: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles")

    # --------------------------------------------------
    # Topology cleanup (mandatory)
    # --------------------------------------------------
    logger.info("Removing degenerate / duplicated / non-manifold geometry")
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()

    logger.info(f"After topology cleanup: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles")

    # --------------------------------------------------
    # Remove small disconnected components (configurable)
    # --------------------------------------------------
    min_tris: int = int(config.get("mesh_min_component_triangles", 500))
    logger.info(f"Removing components smaller than {min_tris} triangles")

    triangle_clusters, cluster_n_tris, _ = mesh.cluster_connected_triangles()
    cluster_n_tris = list(cluster_n_tris)
    remove_triangles = [i for i, c in enumerate(triangle_clusters) if cluster_n_tris[c] < min_tris]
    mesh.remove_triangles_by_index(remove_triangles)
    mesh.remove_unreferenced_vertices()

    logger.info(f"After component filtering: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles")

    # --------------------------------------------------
    # Optional Laplacian smoothing (configurable)
    # --------------------------------------------------
    smoothing_enabled: bool = bool(config.get("mesh_smoothing", False))
    smoothing_iterations: int = int(config.get("mesh_smoothing_iterations", 5))
    if smoothing_enabled:
        logger.info(f"Applying Laplacian smoothing (iterations={smoothing_iterations})")
        mesh = mesh.filter_smooth_laplacian(smoothing_iterations)
        mesh.compute_vertex_normals()

    # --------------------------------------------------
    # Optional quadric decimation (configurable)
    # --------------------------------------------------
    decimation_ratio: Optional[float] = config.get("mesh_decimation_ratio")
    if decimation_ratio is not None:
        ratio = float(decimation_ratio)
        if not (0.0 < ratio < 1.0):
            logger.warning("mesh_decimation_ratio must be between 0 and 1 — skipping decimation")
        else:
            target_tris = int(len(mesh.triangles) * ratio)
            logger.info(f"Applying quadric decimation (ratio={ratio}, target_tris={target_tris:,})")
            mesh = mesh.simplify_quadric_decimation(target_tris)
            mesh.compute_vertex_normals()

    # --------------------------------------------------
    # Save cleaned mesh
    # --------------------------------------------------
    logger.info("Writing cleaned mesh to disk")
    o3d.io.write_triangle_mesh(str(mesh_out), mesh, write_ascii=False)

    report.update({
        "output_vertices": len(mesh.vertices),
        "output_triangles": len(mesh.triangles),
        "min_component_triangles": min_tris,
        "smoothing_enabled": smoothing_enabled,
        "smoothing_iterations": smoothing_iterations if smoothing_enabled else 0,
        "decimation_ratio": decimation_ratio,
    })

    with open(report_out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Mesh post-processing completed successfully")
    logger.info(f"Cleaned mesh saved to: {mesh_out}")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="MARK-2 Mesh Post-Processing")
    parser.add_argument("project_root", type=Path)
    parser.add_argument("--force", action="store_true", help="Overwrite existing cleaned mesh")
    args = parser.parse_args()

    run_mesh_postprocess(args.project_root, args.force)


if __name__ == "__main__":
    main()
