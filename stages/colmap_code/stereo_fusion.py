from pathlib import Path
from core.stage import Stage


class StereoFusionStage(Stage):

    name = "stereo_fusion"

    def run(self, paths, config, logger, tool_runner):

        logger.info("Starting stereo fusion (ultra permissive)")

        output_ply = Path(paths.dense) / "fused.ply"

        cmd = [
            "colmap",
            "stereo_fusion",

            "--workspace_path", str(paths.dense),
            "--workspace_format", "COLMAP",

            "--input_type", "geometric",

            "--output_path", str(output_ply),

            "--StereoFusion.min_num_pixels", "1",
            "--StereoFusion.max_reproj_error", "100",
            "--StereoFusion.max_depth_error", "1",
            "--StereoFusion.max_normal_error", "90",

            "--StereoFusion.check_num_images", "200",

            "--StereoFusion.num_threads", "-1"
        ]

        tool_runner.run(cmd)

        logger.info(f"Dense cloud saved: {output_ply}")