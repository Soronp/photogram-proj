#!/usr/bin/env python3
"""
MARK-2 Dense Reconstruction Stage (OpenMVS)

Responsibilities:
- Run OpenMVS DensifyPointCloud
- Consume scene.mvs produced by openmvs_export
- Require undistorted images folder
- Deterministic, convention-safe, pipeline-compatible
"""

from pathlib import Path
import subprocess

from utils.logger import get_logger
from utils.paths import ProjectPaths


def run_dense(project_root: Path, force: bool):
    """
    Run OpenMVS dense reconstruction.

    User provides a folder that contains:
    - scene.mvs
    - undistorted/ (with images/)
    """
    project_root = Path(project_root).resolve()
    paths = ProjectPaths(project_root)
    paths.ensure_all()

    log = get_logger("dense_reconstruction", paths.root)
    log.info("Starting dense reconstruction (OpenMVS)")

    # --------------------------------------------------
    # Locate OpenMVS working directory
    # --------------------------------------------------
    openmvs_root = paths.openmvs

    scene_file = openmvs_root / "scene.mvs"
    undistorted_dir = openmvs_root / "undistorted"

    if not scene_file.exists() or not undistorted_dir.exists():
        log.warning("Expected OpenMVS layout not found.")
        user_input = input(
            "Please provide the folder containing scene.mvs and undistorted/: "
        ).strip()

        openmvs_root = Path(user_input).resolve()
        scene_file = openmvs_root / "scene.mvs"
        undistorted_dir = openmvs_root / "undistorted"

        if not scene_file.exists():
            raise FileNotFoundError(f"scene.mvs not found in {openmvs_root}")

        if not undistorted_dir.exists():
            raise FileNotFoundError(
                f"undistorted/ folder not found in {openmvs_root}"
            )

    log.info(f"Using scene file: {scene_file}")
    log.info(f"Using undistorted images: {undistorted_dir}")

    # --------------------------------------------------
    # Prepare dense output
    # --------------------------------------------------
    paths.dense.mkdir(parents=True, exist_ok=True)
    fused_file = paths.dense / "fused.ply"

    if fused_file.exists() and not force:
        log.info("Dense output already exists and force is not set; skipping.")
        return

    # --------------------------------------------------
    # OpenMVS DensifyPointCloud
    # --------------------------------------------------
    cmd = [
        "DensifyPointCloud",
        "-i", str(scene_file),
        "-o", str(fused_file),
        "--working-folder", str(openmvs_root),
        "--resolution-level", "1",
        "--max-resolution", "2560",
        "--min-resolution", "640",
        "--number-views", "8",
        "--number-views-fuse", "3",
        "--estimate-colors", "2",
        "--estimate-normals", "2",
        "--filter-point-cloud", "1"
    ]

    log.info(f"[RUN:DensifyPointCloud] {' '.join(cmd)}")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError("DensifyPointCloud failed")

    log.info(f"Dense reconstruction completed: {fused_file}")


# --------------------------------------------------
# Standalone execution
# --------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="MARK-2 Dense Reconstruction (OpenMVS)"
    )
    parser.add_argument(
        "--project", type=str, required=True,
        help="Path to project root"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force re-run even if output exists"
    )

    args = parser.parse_args()
    run_dense(Path(args.project), args.force)
