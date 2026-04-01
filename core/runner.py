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

# DENSE
from stages.dense.colmap.image_undistorter import run as undistort
from stages.dense.colmap.patch_match_stereo import run as patch_match
from stages.dense.colmap.stereo_fusion import run as stereo_fusion

# OPENMVS
from stages.openmvs.export_openmvs import run as openmvs_export
from stages.openmvs.densify import run as openmvs_densify
from stages.openmvs.mesh import run as openmvs_mesh
from stages.openmvs.texture import run as openmvs_texture

# FALLBACK
from stages.mesh.mesh_reconstruction import run as mesh_reconstruction

# GSPLAT
from stages.mesh.gsplat_mesh import run_gsplat_mesh


class PipelineRunner:

    def __init__(self, config, pipeline_type="A"):
        self.pipeline_type = pipeline_type

        project_root = config["paths"]["project_root"]
        self.paths = ProjectPaths(project_root)

        self.logger = setup_logger(self.paths.log_file)
        self.tool_runner = ToolRunner(self.logger)

        self.base_config = deepcopy(config)
        self.config = deepcopy(config)

        self.max_adjustments = 2
        self.adjustment_level = 0

        self._configure_pipeline()
        self._configure_camera_policy()

    # =====================================================
    # INTERNAL EXEC WRAPPER (NEW - NON-BREAKING)
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
    # OUTPUT VALIDATION (NON-INTRUSIVE)
    # =====================================================
    def _check_file(self, path, desc):
        if not path.exists():
            raise RuntimeError(f"{desc} missing: {path}")
        if path.is_file() and path.stat().st_size == 0:
            raise RuntimeError(f"{desc} is empty: {path}")

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
                "mesh": "colmap"
            })

        elif self.pipeline_type == "B":
            self.logger.info("Pipeline B: GLOMAP + COLMAP dense")
            backends.update({
                "sparse": "glomap",
                "dense": "colmap",
                "mesh": "colmap"
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
            self.logger.info("Pipeline E: COLMAP → GSPLAT (no dense)")
            backends.update({
                "sparse": "colmap",
                "dense": None,
                "mesh": "gsplat",
                "texture": None
            })

        else:
            raise ValueError(f"Invalid pipeline type: {self.pipeline_type}")

    # =====================================================
    # CAMERA POLICY
    # =====================================================
    def _configure_camera_policy(self):
        backends = self.config["pipeline"]["backends"]

        camera_model = "OPENCV"

        if backends.get("dense") == "openmvs":
            camera_model = "PINHOLE"

        if backends["sparse"] == "openmvg":
            camera_model = "PINHOLE"

        self.config["pipeline"]["camera_model"] = camera_model
        self.logger.info(f"Camera model set to: {camera_model}")

    # =====================================================
    # MAIN RUN (HARDENED)
    # =====================================================
    def run(self):
        self.logger.info("========== PIPELINE START ==========")

        try:
            self._run_ingestion()

            # -------------------------------
            # SPARSE
            # -------------------------------
            sparse_backend = self.config["pipeline"]["backends"]["sparse"]

            if sparse_backend == "openmvg":
                success = self._execute_stage(
                    "OPENMVG PIPELINE",
                    self._run_openmvg_pipeline
                )

                if not success and self.config["sparse"]["fallback_to_colmap"]:
                    self.logger.warning("⚠️ OpenMVG failed → fallback to COLMAP")
                    self.config["pipeline"]["backends"]["sparse"] = "colmap"
                    self._run_feature_pipeline()
                    self._run_sparse_loop()
            else:
                self._run_feature_pipeline()
                self._run_sparse_loop()

            # -------------------------------
            # GSPLAT SHORT PATH
            # -------------------------------
            if self.pipeline_type == "E":
                self._execute_stage("GSPLAT", self._run_gsplat)
                self.logger.info("========== PIPELINE END ==========")
                return

            # -------------------------------
            # DENSE
            # -------------------------------
            self._execute_stage("DENSE", self._run_dense)

            # Validate dense output
            fused = self.paths.dense / "fused.ply"
            if fused.exists():
                self._check_file(fused, "fused point cloud")

            # -------------------------------
            # MESH
            # -------------------------------
            self._execute_stage("MESH", self._run_mesh)

            # -------------------------------
            # TEXTURE
            # -------------------------------
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

        if self.config.get("downsampling", {}).get("enabled", False):
            self._execute_stage("DOWNSAMPLE", downsample_images, self.paths, self.config, self.logger)

    # =====================================================
    # OPENMVG
    # =====================================================
    def _run_openmvg_pipeline(self):
        result = openmvg_reconstruction(
            self.paths.run_root,
            self.paths.project_root,
            force=False,
            logger=self.logger
        )

        sfm_bin = result.get("sfm_data")

        if not sfm_bin or not sfm_bin.exists():
            return False

        sfm_json = result["matches_dir"] / "sfm_data.json"

        with open(sfm_json) as f:
            data = json.load(f)

        return len(data.get("intrinsics", [])) > 0

    # =====================================================
    # SPARSE
    # =====================================================
    def _run_feature_pipeline(self):
        self._execute_stage("FEATURE EXTRACTION", feature_extraction, self.paths, self.config, self.logger, self.tool_runner)
        self._execute_stage("MATCHING", matching, self.paths, self.config, self.logger, self.tool_runner)

    def _run_sparse_loop(self):
        for i in range(self.max_adjustments + 1):
            self.logger.info(f"===== SPARSE PASS {i+1} =====")

            self._execute_stage("MAPPER", mapper, self.paths, self.config, self.logger, self.tool_runner)

            if self._evaluate_sparse() == "good":
                self.logger.info("✅ Sparse stable")
                return

            if i < self.max_adjustments:
                self._increase_quality()

    def _evaluate_sparse(self):
        pts = self.paths.sparse_model / "points3D.bin"
        return "good" if pts.exists() and pts.stat().st_size > 5_000_000 else "too_low"

    def _increase_quality(self):
        self.adjustment_level += 1
        self.logger.warning(f"⚙️ Increasing quality → {self.adjustment_level}")

    # =====================================================
    # GSPLAT
    # =====================================================
    def _run_gsplat(self):
        run_gsplat_mesh(self.paths, self.config)

    # =====================================================
    # DENSE
    # =====================================================
    def _run_dense(self):
        backend = self.config["pipeline"]["backends"]["dense"]

        if backend is None:
            return

        if backend == "colmap":
            self._execute_stage("UNDISTORT", undistort, self.paths, self.config, self.logger, self.tool_runner)
            self._execute_stage("PATCH MATCH", patch_match, self.paths, self.config, self.logger, self.tool_runner)
            self._execute_stage("STEREO FUSION", stereo_fusion, self.paths, self.config, self.logger, self.tool_runner)

        elif backend == "openmvs":
            self._execute_stage("UNDISTORT", undistort, self.paths, self.config, self.logger, self.tool_runner)
            self._execute_stage("OPENMVS EXPORT", openmvs_export, self.paths, self.config, self.logger, self.tool_runner)
            self._execute_stage("OPENMVS DENSIFY", openmvs_densify, self.paths, self.config, self.logger, self.tool_runner)

    # =====================================================
    # MESH
    # =====================================================
    def _run_mesh(self):
        backend = self.config["pipeline"]["backends"]["mesh"]

        self.logger.info(f"[runner] mesh backend: {backend}")

        if backend == "gsplat":
            return

        if backend == "openmvs":
            self._execute_stage("OPENMVS MESH", openmvs_mesh, self.paths, self.config, self.logger, self.tool_runner)
        else:
            self._execute_stage("MESH",mesh_reconstruction,self.paths,self.config,self.logger, self.tool_runner)

    # =====================================================
    # TEXTURE
    # =====================================================
    def _run_texture(self):
        backend = self.config["pipeline"]["backends"].get("texture")

        if backend == "openmvs":
            self._execute_stage("OPENMVS TEXTURE", openmvs_texture, self.paths, self.config, self.logger, self.tool_runner)