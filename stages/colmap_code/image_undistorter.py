from pathlib import Path
from core.stage import Stage


class ImageUndistorterStage(Stage):

    name = "image_undistorter"

    def run(self, paths, config, logger, tool_runner):

        logger.info("Starting image undistortion")

        # --------------------------------------------------
        # Resolve sparse model directory
        # --------------------------------------------------

        sparse_root = Path(paths.sparse)

        sparse_model = sparse_root / "0"

        if not sparse_model.exists():
            raise RuntimeError(
                f"Sparse model not found at {sparse_model}. "
                "Mapper likely failed or produced no model."
            )

        logger.info(f"Using sparse model: {sparse_model}")

        # --------------------------------------------------
        # Prepare dense workspace
        # --------------------------------------------------

        dense_workspace = Path(paths.dense)
        dense_workspace.mkdir(parents=True, exist_ok=True)

        # --------------------------------------------------
        # COLMAP image undistorter command
        # --------------------------------------------------

        cmd = [
            "colmap",
            "image_undistorter",

            "--image_path",
            str(paths.images),

            "--input_path",
            str(sparse_model),

            "--output_path",
            str(dense_workspace),

            "--output_type",
            "COLMAP",

            "--max_image_size",
            "2048",
        ]

        tool_runner.run(cmd)

        logger.info("Image undistortion finished")