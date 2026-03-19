from core.stage import Stage


class PatchMatchStereoStage(Stage):

    name = "patch_match_stereo"

    def run(self, paths, config, logger, tool_runner):

        logger.info("Starting PatchMatch stereo")

        cmd = [
            "colmap",
            "patch_match_stereo",

            "--workspace_path", str(paths.dense),
            "--workspace_format", "COLMAP",

            # Resolution control
            "--PatchMatchStereo.max_image_size", "2048",

            # Sampling / optimization
            "--PatchMatchStereo.window_radius", "5",
            "--PatchMatchStereo.num_samples", "15",
            "--PatchMatchStereo.num_iterations", "5",

            # Geometry consistency
            "--PatchMatchStereo.geom_consistency", "1",

            # Filtering (balanced)
            "--PatchMatchStereo.filter", "1",
            "--PatchMatchStereo.filter_min_ncc", "0.1",
            "--PatchMatchStereo.filter_min_triangulation_angle", "3",
            "--PatchMatchStereo.filter_min_num_consistent", "2",
            "--PatchMatchStereo.filter_geom_consistency_max_cost", "1",

            # Stability
            "--PatchMatchStereo.allow_missing_files", "0",

            # Debug (optional)
            "--PatchMatchStereo.write_consistency_graph", "0"
        ]

        tool_runner.run(cmd)

        logger.info("PatchMatch finished")