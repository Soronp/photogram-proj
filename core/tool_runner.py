import subprocess
import time


class ToolRunner:
    """
    Executes external tools such as COLMAP
    with controlled logging and error handling.
    """

    def __init__(self, logger):
        self.logger = logger

    def run(
        self,
        command,
        cwd=None,
        env=None,
        check=True,
        capture_output=False
    ):
        """
        Run external command.

        Parameters
        ----------
        command : list[str]
        cwd : Path | None
        env : dict | None
        check : bool
        capture_output : bool
        """

        cmd_str = " ".join(map(str, command))
        self.logger.info(f"Running command: {cmd_str}")

        start_time = time.time()

        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        collected_output = []

        for line in process.stdout:
            line = line.rstrip()

            if capture_output:
                collected_output.append(line)

            self.logger.info(line)

        process.wait()

        runtime = time.time() - start_time

        if process.returncode != 0:

            self.logger.error(
                f"Command failed with exit code {process.returncode}"
            )

            if check:
                raise RuntimeError(f"Command failed: {cmd_str}")

        self.logger.info(
            f"Command finished in {runtime:.2f} seconds"
        )

        if capture_output:
            return "\n".join(collected_output)

        return process.returncode