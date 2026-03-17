from core.stage import Stage


class FeatureExtractionStage(Stage):

    name = "feature_extraction"

    def run(self, paths, config, logger, tool_runner):

        logger.info("Starting COLMAP feature extraction")

        cmd = [
            "colmap",
            "feature_extractor",

            "--database_path", str(paths.database),
            "--image_path", str(paths.images),

            "--FeatureExtraction.use_gpu", "1",
            "--FeatureExtraction.num_threads", "-1",

            "--SiftExtraction.max_image_size", "2048",
            "--SiftExtraction.max_num_features", "20000",

            "--SiftExtraction.estimate_affine_shape", "1",
            "--SiftExtraction.domain_size_pooling", "1",

            "--ImageReader.single_camera", "0"
        ]

        tool_runner.run(cmd)

        logger.info("Feature extraction finished")