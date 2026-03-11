#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import open3d as o3d
import shutil


# --------------------------------------------------
# Detect mesh output
# --------------------------------------------------

def detect_mesh(openmvs_root: Path):

    candidates = [

        "fused_mesh.ply",
        "mesh.ply",
        "scene_mesh.ply",
        "model_dense_mesh.ply"
    ]

    for name in candidates:

        p = openmvs_root / name

        if p.exists() and p.stat().st_size > 10000:
            return p

    return None


# --------------------------------------------------
# Run ReconstructMesh
# --------------------------------------------------

def run_reconstruct_mesh(paths, logger, tools):

    openmvs_root = paths.openmvs
    fused = openmvs_root / "fused.mvs"

    if not fused.exists():
        raise RuntimeError("fused.mvs missing")

    logger.info("[mesh] running ReconstructMesh")

    tools.run(
        "openmvs.reconstructmesh",
        [
            "--input-file", "fused.mvs",
            "--verbosity", "2"
        ],
        cwd=openmvs_root
    )

    mesh = detect_mesh(openmvs_root)

    if mesh is None:
        raise RuntimeError("ReconstructMesh produced no mesh")

    logger.info(f"[mesh] detected mesh → {mesh.name}")

    return mesh


# --------------------------------------------------
# Fix normals
# --------------------------------------------------

def fix_normals(mesh, logger):

    mesh.compute_vertex_normals()

    normals = np.asarray(mesh.vertex_normals)
    verts = np.asarray(mesh.vertices)

    center = verts.mean(axis=0)

    dot = np.einsum("ij,ij->i", normals, verts - center)

    inward_ratio = np.sum(dot < 0) / len(dot)

    logger.info(f"[mesh] inward normals ratio: {inward_ratio:.3f}")

    if inward_ratio > 0.1:

        normals *= -1
        mesh.vertex_normals = o3d.utility.Vector3dVector(normals)
        mesh.compute_vertex_normals()

        logger.info("[mesh] normals flipped")

    return mesh


# --------------------------------------------------
# Remove small clusters
# --------------------------------------------------

def remove_small_clusters(mesh, logger):

    clusters, counts, _ = mesh.cluster_connected_triangles()

    clusters = np.asarray(clusters)
    counts = np.asarray(counts)

    keep = counts >= 50

    mask = np.array([keep[c] for c in clusters])

    mesh.remove_triangles_by_mask(~mask)
    mesh.remove_unreferenced_vertices()

    logger.info("[mesh] small clusters removed")

    return mesh


# --------------------------------------------------
# Stage entry
# --------------------------------------------------

def run(paths, tools, config, logger):

    logger.info("[gen_mesh] starting")

    openmvs_root = paths.openmvs
    mesh_dir = paths.mesh

    mesh_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # Reconstruction
    # --------------------------------------------------

    mesh_file = run_reconstruct_mesh(paths, logger, tools)

    # --------------------------------------------------
    # Copy raw mesh
    # --------------------------------------------------

    mesh_raw = mesh_dir / "mesh_raw.ply"

    shutil.copy(mesh_file, mesh_raw)

    logger.info(f"[mesh] raw mesh copied → {mesh_raw}")

    # --------------------------------------------------
    # Open3D cleanup
    # --------------------------------------------------

    mesh = o3d.io.read_triangle_mesh(str(mesh_raw))

    logger.info(
        f"[mesh] loaded {len(mesh.vertices)} verts, {len(mesh.triangles)} tris"
    )

    mesh = fix_normals(mesh, logger)

    mesh = remove_small_clusters(mesh, logger)

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_unreferenced_vertices()

    mesh.compute_vertex_normals()

    mesh_clean = mesh_dir / "mesh_cleaned.ply"

    o3d.io.write_triangle_mesh(str(mesh_clean), mesh)

    logger.info(f"[mesh] cleaned mesh saved → {mesh_clean}")

    logger.info("[gen_mesh] completed")