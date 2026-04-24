from copy import deepcopy
import json
import traceback

from utils.paths import ProjectPaths
from utils.logger import setup_logger
from core.tool_runner import ToolRunner

# INGESTION
from stages.ingestion.ingest_images import run as ingest_images
from stages.ingestion.validate_images import run as validate_images
from stages.ingestion.downsample import run as downsample_images

# SPARSE
from stages.sparse.feature_extraction import run as feature_extraction
from stages.sparse.feature_matching import run as matching
from stages.sparse.mapper import run as mapper

# OPENMVG
from stages.sparse.openmvg_reconstruction import run as openmvg_reconstruction

# DENSE (COLMAP)
from stages.dense.colmap.image_undistorter import run as undistort
from stages.dense.colmap.patch_match_stereo import run as patch_match
from stages.dense.colmap.stereo_fusion import run as stereo_fusion

# OPENMVS
from stages.openmvs.export_openmvs import run as openmvs_export
from stages.openmvs.densify import run as openmvs_densify
from stages.openmvs.mesh import run as openmvs_mesh
from stages.openmvs.texture import run as openmvs_texture

# GENERIC MESH
from stages.mesh.mesh_reconstruction import run as mesh_reconstruction

# NERF
from stages.nerf.pipeline import run_nerfstudio_dense


class PipelineRunner:

    def __init__(self, config, pipeline_type="A"):
        self.pipeline_type = pipeline_type

        self.paths = ProjectPaths(config["paths"]["project_root"])
        self.logger = setup_logger(self.paths.log_file)
        self.tool_runner = ToolRunner(self.logger)

        self.base_config = deepcopy(config)
        self.config = deepcopy(config)

        self._configure_pipeline()
        self._configure_camera_policy()

    # =====================================================
    # EXECUTION WRAPPER
    # =====================================================
    def _execute_stage(self, name, fn, *args):
        self.logger.info(f"===== START: {name} =====")
        try:
            result = fn(*args)
            self.logger.info(f"===== END: {name} =====")
            return result
        except Exception as e:
            self.logger.error(f"❌ FAILED: {name} → {e}")
            self.logger.error(traceback.format_exc())
            raise

    # =====================================================
    # FILE CHECK
    # =====================================================
    def _check_file(self, path, desc):
        if not path.exists():
            raise RuntimeError(f"{desc} missing: {path}")
        if path.is_file() and path.stat().st_size == 0:
            raise RuntimeError(f"{desc} is empty: {path}")

    # =====================================================
    # BACKEND-AWARE DENSE VALIDATION (🔥 FIX)
    # =====================================================
    def _validate_dense_output(self):
        backend = self.config["pipeline"]["backends"]["dense"]

        if backend == "colmap":
            fused = self.paths.dense / "fused.ply"
            self._check_file(fused, "COLMAP fused point cloud")

        elif backend == "openmvs":
            mvs = self.paths.run_root / "openmvs" / "scene_dense.mvs"
            self._check_file(mvs, "OpenMVS dense scene")

        elif backend == "nerfstudio":
            fused = self.paths.dense / "fused.ply"

            if fused.exists():
                self._check_file(fused, "Nerfstudio point cloud")
            else:
                self.logger.warning(
                    "[DENSE] Nerfstudio output not standardized (no fused.ply found)"
                )

        else:
            raise RuntimeError(f"Unknown dense backend: {backend}")

    # =====================================================
    # PIPELINE CONFIG
    # =====================================================
    def _configure_pipeline(self):
        backends = self.config["pipeline"]["backends"]

        if self.pipeline_type == "A":
            self.logger.info("Pipeline A: COLMAP full")
            backends.update({
                "sparse": "colmap",
                "dense": "colmap",
                "mesh": "colmap",
                "texture": "colmap"
            })

        elif self.pipeline_type == "C":
            self.logger.info("Pipeline C: COLMAP → OpenMVS")
            backends.update({
                "sparse": "colmap",
                "dense": "openmvs",
                "mesh": "openmvs",
                "texture": "openmvs"
            })

        elif self.pipeline_type == "D":
            self.logger.info("Pipeline D: OpenMVG → OpenMVS")
            backends.update({
                "sparse": "openmvg",
                "dense": "openmvs",
                "mesh": "openmvs",
                "texture": "openmvs"
            })

        elif self.pipeline_type == "E":
            self.logger.info("Pipeline E: COLMAP → Nerfstudio")
            backends.update({
                "sparse": "colmap",
                "dense": "nerfstudio",
                "mesh": "colmap"
            })

        else:
            raise ValueError(f"Invalid pipeline type: {self.pipeline_type}")

    # =====================================================
    # CAMERA POLICY
    # =====================================================
    def _configure_camera_policy(self):
        backends = self.config["pipeline"]["backends"]

        if backends["dense"] == "openmvs":
            camera_model = "PINHOLE"
        elif backends["sparse"] == "openmvg":
            camera_model = "PINHOLE"
        else:
            camera_model = "OPENCV"

        self.config["pipeline"]["camera_model"] = camera_model
        self.logger.info(f"Camera model set to: {camera_model}")

    # =====================================================
    # MAIN RUN
    # =====================================================
    def run(self):
        self.logger.info("========== PIPELINE START ==========")

        try:
            self._run_ingestion()
            self._run_sparse()

            # -----------------------
            # DENSE
            # -----------------------
            self._execute_stage("DENSE", self._run_dense)
            self._validate_dense_output()   # 🔥 FIXED

            # -----------------------
            # MESH
            # -----------------------
            self._execute_stage("MESH", self._run_mesh)

            # -----------------------
            # TEXTURE
            # -----------------------
            self._execute_stage("TEXTURE", self._run_texture)

        except Exception as e:
            self.logger.error(f"❌ PIPELINE FAILED: {e}")
            raise

        self.logger.info("========== PIPELINE END ==========")

    # =====================================================
    # INGESTION
    # =====================================================
    def _run_ingestion(self):
        self._execute_stage("INGEST", ingest_images, self.paths, self.config, self.logger)
        self._execute_stage("VALIDATE", validate_images, self.paths, self.config, self.logger)

        if self.config["downsampling"]["enabled"]:
            self._execute_stage("DOWNSAMPLE", downsample_images, self.paths, self.config, self.logger)

    # =====================================================
    # SPARSE
    # =====================================================
    def _run_sparse(self):
        backend = self.config["pipeline"]["backends"]["sparse"]

        if backend == "openmvg":
            self._execute_stage("OPENMVG", openmvg_reconstruction,
                                self.paths.run_root, self.paths.project_root, False, self.logger)
        else:
            self._execute_stage("FEATURE EXTRACTION", feature_extraction,
                                self.paths, self.config, self.logger, self.tool_runner)
            self._execute_stage("MATCHING", matching,
                                self.paths, self.config, self.logger, self.tool_runner)
            self._execute_stage("MAPPER", mapper,
                                self.paths, self.config, self.logger, self.tool_runner)

    # =====================================================
    # DENSE
    # =====================================================
    def _run_dense(self):
        backend = self.config["pipeline"]["backends"]["dense"]

        if backend == "colmap":
            self._execute_stage("UNDISTORT", undistort, self.paths, self.config, self.logger, self.tool_runner)
            self._execute_stage("PATCH MATCH", patch_match, self.paths, self.config, self.logger, self.tool_runner)
            self._execute_stage("STEREO FUSION", stereo_fusion, self.paths, self.config, self.logger, self.tool_runner)

        elif backend == "openmvs":
            self._execute_stage("UNDISTORT", undistort, self.paths, self.config, self.logger, self.tool_runner)
            self._execute_stage("OPENMVS EXPORT", openmvs_export, self.paths, self.config, self.logger, self.tool_runner)
            self._execute_stage("OPENMVS DENSIFY", openmvs_densify, self.paths, self.config, self.logger, self.tool_runner)

        elif backend == "nerfstudio":
            self._execute_stage("NERFSTUDIO", run_nerfstudio_dense,
                                self.paths, self.config, self.logger, self.tool_runner)

    # =====================================================
    # MESH
    # =====================================================
    def _run_mesh(self):
        backend = self.config["pipeline"]["backends"]["mesh"]

        if backend == "openmvs":
            self._execute_stage("OPENMVS MESH", openmvs_mesh,
                                self.paths, self.config, self.logger, self.tool_runner)
        else:
            self._execute_stage("MESH", mesh_reconstruction,
                                self.paths, self.config, self.logger, self.tool_runner)

    # =====================================================
    # TEXTURE
    # =====================================================
    def _run_texture(self):
        backend = self.config["pipeline"]["backends"].get("texture")

        if backend == "openmvs":
            self._execute_stage("OPENMVS TEXTURE", openmvs_texture,
                                self.paths, self.config, self.logger, self.tool_runner)