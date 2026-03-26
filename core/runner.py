from copy import deepcopy
import json

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
        self.adjustment_level = 0  # 🔥 TRACK QUALITY LEVEL

        self._configure_pipeline()
        self._configure_camera_policy()

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

        else:
            raise ValueError(f"Invalid pipeline type: {self.pipeline_type}")

    # =====================================================
    # CAMERA POLICY
    # =====================================================
    def _configure_camera_policy(self):
        backends = self.config["pipeline"]["backends"]

        camera_model = "OPENCV"
        if backends["dense"] == "openmvs":
            camera_model = "PINHOLE"
        if backends["sparse"] == "openmvg":
            camera_model = "PINHOLE"

        self.config["pipeline"]["camera_model"] = camera_model
        self.logger.info(f"Camera model set to: {camera_model}")

    # =====================================================
    # MAIN RUN
    # =====================================================
    def run(self):
        self.logger.info("========== PIPELINE START ==========")

        self._run_ingestion()

        sparse_backend = self.config["pipeline"]["backends"]["sparse"]

        if sparse_backend == "openmvg":
            success = self._run_openmvg_pipeline()

            if not success and self.config["sparse"]["fallback_to_colmap"]:
                self.logger.warning("⚠️ OpenMVG failed → falling back to COLMAP")
                self.config["pipeline"]["backends"]["sparse"] = "colmap"
                self._run_feature_pipeline()
                self._run_sparse_loop()
        else:
            self._run_feature_pipeline()
            self._run_sparse_loop()

        self._run_dense()
        self._run_mesh()
        self._run_texture()

        self.logger.info("========== PIPELINE END ==========")

    # =====================================================
    # INGESTION
    # =====================================================
    def _run_ingestion(self):
        ingest_images(self.paths, self.config, self.logger)
        validate_images(self.paths, self.config, self.logger)

        if self.config.get("downsampling", {}).get("enabled", False):
            downsample_images(self.paths, self.config, self.logger)

    # =====================================================
    # OPENMVG PIPELINE
    # =====================================================
    def _run_openmvg_pipeline(self):
        self.logger.info("===== OPENMVG PIPELINE =====")

        result = openmvg_reconstruction(
            self.paths.run_root,
            self.paths.project_root,
            force=False,
            logger=self.logger
        )

        sfm_bin = result.get("sfm_data")

        if not sfm_bin or not sfm_bin.exists():
            self.logger.error("❌ OpenMVG failed: no sfm_data.bin")
            return False

        sfm_json = result["matches_dir"] / "sfm_data.json"

        with open(sfm_json) as f:
            data = json.load(f)

        if len(data.get("intrinsics", [])) == 0:
            self.logger.error("❌ OpenMVG failed: NO INTRINSICS")
            return False

        self.logger.info("✅ OpenMVG reconstruction successful")
        return True

    # =====================================================
    # SPARSE
    # =====================================================
    def _run_feature_pipeline(self):
        feature_extraction(self.paths, self.config, self.logger, self.tool_runner)
        matching(self.paths, self.config, self.logger, self.tool_runner)

    def _run_sparse_loop(self):
        for i in range(self.max_adjustments + 1):
            self.logger.info(f"===== SPARSE PASS {i+1} =====")

            mapper(self.paths, self.config, self.logger, self.tool_runner)

            status = self._evaluate_sparse()

            if status == "good":
                self.logger.info("✅ Sparse reconstruction stable")
                return

            if i < self.max_adjustments:
                self._increase_quality()
            else:
                self.logger.warning("⚠️ Sparse did not fully stabilize")

    def _evaluate_sparse(self):
        pts_file = self.paths.sparse_model / "points3D.bin"
        if not pts_file.exists():
            return "too_low"

        size = pts_file.stat().st_size
        return "good" if size > 5_000_000 else "too_low"

    # =====================================================
    # 🔥 QUALITY ESCALATION (FIXED)
    # =====================================================
    def _increase_quality(self):
        self.adjustment_level += 1
        self.logger.warning(f"⚙️ Increasing quality level → {self.adjustment_level}")

        sparse_cfg = self.config.setdefault("sparse", {})
        dense_cfg = self.config.setdefault("dense", {})

        if self.adjustment_level == 1:
            # 🔥 Improve matching robustness
            sparse_cfg["max_num_features"] = 12000
            sparse_cfg["min_num_matches"] = 30

        elif self.adjustment_level == 2:
            # 🔥 Improve geometry stability
            dense_cfg["PatchMatchStereo.num_iterations"] = 7
            dense_cfg["PatchMatchStereo.window_radius"] = 7
            dense_cfg["PatchMatchStereo.filter_min_num_consistent"] = 3
            dense_cfg["PatchMatchStereo.geom_consistency"] = 1

        else:
            self.logger.warning("Max quality escalation reached")

    # =====================================================
    # DENSE
    # =====================================================
    def _run_dense(self):
        backend = self.config["pipeline"]["backends"]["dense"]

        if backend == "colmap":
            undistort(self.paths, self.config, self.logger, self.tool_runner)

            # 🔥 enforce deterministic patch match behavior
            self.config["dense"].update({
                "PatchMatchStereo.geom_consistency": 1,
                "PatchMatchStereo.num_iterations": 5,
                "PatchMatchStereo.cache_size": 64
            })

            patch_match(self.paths, self.config, self.logger, self.tool_runner)
            stereo_fusion(self.paths, self.config, self.logger, self.tool_runner)

        elif backend == "openmvs":
            if self.pipeline_type == "C":
                undistort(self.paths, self.config, self.logger, self.tool_runner)

            openmvs_export(self.paths, self.config, self.logger, self.tool_runner)
            openmvs_densify(self.paths, self.config, self.logger, self.tool_runner)

    # =====================================================
    # MESH
    # =====================================================
    def _run_mesh(self):
        backend = self.config["pipeline"]["backends"]["mesh"]

        if backend == "openmvs":
            openmvs_mesh(self.paths, self.config, self.logger, self.tool_runner)
        else:
            mesh_reconstruction(self.paths, self.config, self.logger)

    # =====================================================
    # TEXTURE
    # =====================================================
    def _run_texture(self):
        backend = self.config["pipeline"]["backends"].get("texture")

        if backend == "openmvs":
            openmvs_texture(self.paths, self.config, self.logger, self.tool_runner)