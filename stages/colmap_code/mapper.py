from pathlib import Path
from core.stage import Stage


class MapperStage(Stage):

    name = "mapper"

    def run(self, paths, config, logger, tool_runner):

        logger.info("Starting sparse reconstruction")

        sparse_root = Path(paths.sparse)

        sparse_root.mkdir(parents=True, exist_ok=True)

        cmd = [
            "colmap",
            "mapper",

            "--database_path", str(paths.database),
            "--image_path", str(paths.images),

            "--output_path", str(sparse_root),

            "--Mapper.num_threads", "-1",

            "--Mapper.init_min_num_inliers", "50",
            "--Mapper.ba_global_max_num_iterations", "50",
        ]

        tool_runner.run(cmd)

        model = sparse_root / "0"

        if not model.exists():
            raise RuntimeError("Sparse model not produced")

        logger.info(f"Sparse model created: {model}")