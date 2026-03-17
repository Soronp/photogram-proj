from core.stage import Stage


class PatchMatchStereoStage(Stage):

    name = "patch_match_stereo"

    def run(self, paths, config, logger, tool_runner):

        logger.info("Starting PatchMatch stereo (ultra loose settings)")

        cmd = [
            "colmap",
            "patch_match_stereo",

            "--workspace_path", str(paths.dense),
            "--workspace_format", "COLMAP",

            # Use GPU automatically
            "--PatchMatchStereo.gpu_index", "0",

            # Resolution
            "--PatchMatchStereo.max_image_size", "2048",

            # Sampling
            "--PatchMatchStereo.window_radius", "5",
            "--PatchMatchStereo.num_samples", "30",
            "--PatchMatchStereo.num_iterations", "7",

            # Geometry
            "--PatchMatchStereo.geom_consistency", "1",

            # Ultra-loose filters
            "--PatchMatchStereo.filter", "1",
            "--PatchMatchStereo.filter_min_ncc", "-1",
            "--PatchMatchStereo.filter_min_triangulation_angle", "0.1",
            "--PatchMatchStereo.filter_min_num_consistent", "1",
            "--PatchMatchStereo.filter_geom_consistency_max_cost", "100",

            # Allow incomplete data
            "--PatchMatchStereo.allow_missing_files", "1",

            # Debugging / stronger fusion later
            "--PatchMatchStereo.write_consistency_graph", "1"
        ]

        tool_runner.run(cmd)

        logger.info("PatchMatch finished")