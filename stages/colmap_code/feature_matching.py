from core.stage import Stage


class FeatureMatchingStage(Stage):

    name = "feature_matching"

    def run(self, paths, config, logger, tool_runner):

        logger.info("Starting COLMAP exhaustive matching")

        cmd = [
            "colmap",
            "exhaustive_matcher",

            "--database_path", str(paths.database),

            "--FeatureMatching.use_gpu", "1",
            "--FeatureMatching.num_threads", "-1",

            "--FeatureMatching.max_num_matches", "65536",

            "--SiftMatching.max_ratio", "0.8",

            "--TwoViewGeometry.min_num_inliers", "12"
        ]

        tool_runner.run(cmd)

        logger.info("Feature matching finished")