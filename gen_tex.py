#!/usr/bin/env python3
"""
gen_tex.py

MARK-2 Texture Mapping Stage
----------------------------
- Attaches appearance to cleaned mesh
- Generates UV atlas if supported by Open3D
- Deterministic, restart-safe
- Input : mesh/mesh_cleaned.ply
- Output: textures/textured_mesh.obj
"""

import argparse
import json
import shutil
from pathlib import Path

import open3d as o3d

from utils.logger import get_logger
from utils.paths import ProjectPaths
from utils.config import load_config


# ------------------------------------------------------------------
# Texture mapping pipeline
# ------------------------------------------------------------------
def run_texture_mapping(project_root: Path, force: bool):
    paths = ProjectPaths(project_root)
    config = load_config(project_root)
    logger = get_logger("texture_mapping", project_root)

    mesh_in: Path = paths.mesh / "mesh_cleaned.ply"
    tex_dir: Path = paths.textures
    mesh_out: Path = tex_dir / "textured_mesh.obj"
    report_out: Path = tex_dir / "texture_report.json"

    logger.info("=== Texture Mapping Stage ===")
    logger.info(f"Mesh input  : {mesh_in}")
    logger.info(f"Texture dir : {tex_dir}")

    if not mesh_in.exists():
        raise FileNotFoundError(f"Cleaned mesh not found: {mesh_in}")

    if tex_dir.exists() and not force:
        logger.info("Textures already exist — skipping (use --force to regenerate)")
        return

    if tex_dir.exists() and force:
        logger.warning("Removing existing textures directory (--force)")
        shutil.rmtree(tex_dir)

    tex_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # Load mesh
    # --------------------------------------------------
    logger.info("Loading cleaned mesh")
    mesh = o3d.io.read_triangle_mesh(str(mesh_in))
    if not mesh.has_triangles():
        raise RuntimeError("Mesh contains no triangles")

    mesh.compute_vertex_normals()
    logger.info(f"Loaded mesh: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles")

    # --------------------------------------------------
    # UV atlas generation (configurable)
    # --------------------------------------------------
    uv_supported: bool = hasattr(mesh, "compute_uvatlas")
    uv_generated: bool = False

    if uv_supported:
        atlas_size: int = int(config.get("texture_atlas_size", 2048))
        gutter: int = int(config.get("texture_gutter", 2))

        logger.info(f"Generating UV atlas (size={atlas_size}, gutter={gutter})")
        mesh = mesh.compute_uvatlas(atlas_size=atlas_size, gutter=gutter)

        if mesh.has_triangle_uvs():
            uv_generated = True
            logger.info("UV atlas generated successfully")
        else:
            logger.warning("UV generation failed — proceeding without UVs")
    else:
        logger.warning(
            "UV atlas generation NOT supported by this Open3D build "
            "(legacy geometry API). Proceeding without UVs."
        )

    # --------------------------------------------------
    # Write textured mesh
    # --------------------------------------------------
    logger.info(f"Writing textured mesh to: {mesh_out}")
    o3d.io.write_triangle_mesh(str(mesh_out), mesh, write_triangle_uvs=uv_generated)

    # --------------------------------------------------
    # Report
    # --------------------------------------------------
    report = {
        "mesh_input": str(mesh_in.name),
        "mesh_output": str(mesh_out.name),
        "vertices": len(mesh.vertices),
        "triangles": len(mesh.triangles),
        "uv_supported": uv_supported,
        "uv_generated": uv_generated,
        "deterministic": True,
        "note": (
            "Photometric texture baking requires Tensor Open3D or "
            "COLMAP texture_mesher integration"
        ),
    }

    with open(report_out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Texture mapping stage completed successfully")
    logger.info(f"Textured mesh saved to: {mesh_out}")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="MARK-2 Texture Mapping")
    parser.add_argument("project_root", type=Path)
    parser.add_argument("--force", action="store_true", help="Overwrite existing texture outputs")
    args = parser.parse_args()

    run_texture_mapping(args.project_root, args.force)


if __name__ == "__main__":
    main()
