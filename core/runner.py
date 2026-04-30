from copy import deepcopy
import traceback
import json
from pathlib import Path

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
from stages.sparse.openmvg_reconstruction import run as openmvg_reconstruction

# COLMAP DENSE
from stages.dense.colmap.image_undistorter import run as undistort
from stages.dense.colmap.patch_match_stereo import run as patch_match
from stages.dense.colmap.stereo_fusion import run as stereo_fusion

# OPENMVS
from stages.openmvs.export_openmvs import run as openmvs_export
from stages.openmvs.densify import run as openmvs_densify
from stages.openmvs.mesh import run as openmvs_mesh
from stages.openmvs.texture import run as openmvs_texture

# GENERIC
from stages.mesh.mesh_reconstruction import run as mesh_reconstruction


class PipelineRunner:

    def __init__(self, config, run_root: Path, pipeline_type="A"):

        self.pipeline_type = pipeline_type.upper()
        self.run_root = Path(run_root).resolve()

        self.paths = ProjectPaths(self.run_root)
        self.logger = setup_logger(self.paths.log_file)
        self.tool_runner = ToolRunner(self.logger)

        self.config = deepcopy(config)

        self.state_path = self.run_root / "pipeline_state.json"
        self.state = self._load_state()
        self._handle_resume_choice()

        self._configure_pipeline()
        self._configure_camera_policy()

    # =====================================================
    # STATE
    # =====================================================
    def _load_state(self):
        if self.state_path.exists():
            try:
                return json.loads(self.state_path.read_text())
            except:
                return {}
        return {}

    def _save_state(self):
        self.state_path.write_text(json.dumps(self.state, indent=2))

    def _handle_resume_choice(self):
        if not self.state:
            return

        print("\nPrevious state detected:")
        print(json.dumps(self.state, indent=2))

        if input("\nResume? (y/n) [y]: ").strip().lower() == "n":
            self.state = {}
            self._save_state()

    # =====================================================
    # EXECUTION
    # =====================================================
    def _execute_stage(self, name, fn, *args):

        if self.state.get(name) == "complete":
            self.logger.info(f"[SKIP] {name}")
            return

        self.logger.info(f"===== START: {name} =====")

        try:
            result = fn(*args)

            self.state[name] = "complete"
            self._save_state()

            self.logger.info(f"===== END: {name} =====")
            return result

        except Exception as e:
            self.state[name] = "failed"
            self._save_state()

            self.logger.error(f"FAILED {name}: {e}")
            self.logger.error(traceback.format_exc())
            raise

    # =====================================================
    # CONFIG
    # =====================================================
    def _configure_pipeline(self):

        mapping = {
            "A": ("colmap", "colmap", "colmap", "colmap"),
            "B": ("glomap", "colmap", "colmap", "colmap"),
            "C": ("colmap", "openmvs", "openmvs", "openmvs"),
            "D": ("openmvg", "openmvs", "openmvs", "openmvs"),
        }

        s, d, m, t = mapping[self.pipeline_type]

        self.config["pipeline"]["backends"].update({
            "sparse": s,
            "dense": d,
            "mesh": m,
            "texture": t
        })

    def _configure_camera_policy(self):
        self.config["pipeline"]["camera_model"] = (
            "PINHOLE"
            if self.config["pipeline"]["backends"]["sparse"] == "openmvg"
            else "OPENCV"
        )

    # =====================================================
    # MAIN
    # =====================================================
    def run(self):

        self.logger.info("========== PIPELINE START ==========")

        try:
            self._run_ingestion()
            self._run_sparse()
            self._run_dense()
            self._run_export()   # 🔥 moved AFTER undistort
            self._run_mesh()
            self._run_texture()

        except Exception as e:
            self.logger.error(f"PIPELINE FAILED: {e}")
            raise

        self.logger.info("========== PIPELINE END ==========")

    # =====================================================
    # INGESTION
    # =====================================================
    def _run_ingestion(self):

        self._execute_stage("INGEST", ingest_images, self.paths, self.config, self.logger)
        self._execute_stage("VALIDATE", validate_images, self.paths, self.config, self.logger)

        if self.config["downsampling"]["enabled"]:
            self._execute_stage("DOWNSAMPLE", downsample_images,
                                self.paths, self.config, self.logger)

    # =====================================================
    # SPARSE
    # =====================================================
    def _run_sparse(self):

        backend = self.config["pipeline"]["backends"]["sparse"]

        if backend == "openmvg":
            self._execute_stage("OPENMVG", openmvg_reconstruction,
                                self.paths.run_root, self.paths.run_root, False, self.logger)
        else:
            self._execute_stage("FEATURE", feature_extraction,
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
            self._execute_stage("UNDISTORT", undistort,
                                self.paths, self.config, self.logger, self.tool_runner)

            self._execute_stage("PATCH MATCH", patch_match,
                                self.paths, self.config, self.logger, self.tool_runner)

            self._execute_stage("STEREO FUSION", stereo_fusion,
                                self.paths, self.config, self.logger, self.tool_runner)

        elif backend == "openmvs":
            # only UNDISTORT needed before export
            if self.config["pipeline"]["backends"]["sparse"] == "colmap":
                self._execute_stage("UNDISTORT", undistort,
                                    self.paths, self.config, self.logger, self.tool_runner)

    # =====================================================
    # 🔥 EXPORT (FIXED POSITION)
    # =====================================================
    def _run_export(self):

        if self.config["pipeline"]["backends"]["dense"] != "openmvs":
            return

        scene = self.paths.openmvs / "scene.mvs"

        if scene.exists():
            self.logger.info("OPENMVS EXPORT: scene.mvs exists → skipping")
            return

        self._execute_stage("OPENMVS EXPORT",
                            openmvs_export,
                            self.paths, self.config, self.logger, self.tool_runner)

    # =====================================================
    # MESH
    # =====================================================
    def _run_mesh(self):

        backend = self.config["pipeline"]["backends"]["mesh"]

        if backend == "openmvs":
            self._execute_stage("OPENMVS DENSIFY",
                                openmvs_densify,
                                self.paths, self.config, self.logger, self.tool_runner)

            self._execute_stage("OPENMVS MESH",
                                openmvs_mesh,
                                self.paths, self.config, self.logger, self.tool_runner)
        else:
            self._execute_stage("MESH",
                                mesh_reconstruction,
                                self.paths, self.config, self.logger, self.tool_runner)

    # =====================================================
    # TEXTURE
    # =====================================================
    def _run_texture(self):

        if self.config["pipeline"]["backends"]["texture"] == "openmvs":
            self._execute_stage("OPENMVS TEXTURE",
                                openmvs_texture,
                                self.paths, self.config, self.logger, self.tool_runner)