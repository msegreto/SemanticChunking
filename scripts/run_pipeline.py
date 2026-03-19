from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

# Force all pipeline runs to use GPU 0.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from src.pipelines.experiment_orchestrator import ExperimentOrchestrator


def _assert_gpu_zero_available() -> None:
    if shutil.which("nvidia-smi") is None:
        raise RuntimeError("GPU 0 required, but 'nvidia-smi' is not available.")

    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"GPU 0 required, but nvidia-smi failed (exit_code={result.returncode})."
        )

    gpu_indexes = {line.strip() for line in result.stdout.splitlines() if line.strip()}
    if "0" not in gpu_indexes:
        raise RuntimeError("GPU 0 required, but it is not available on this machine.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full chunking pipeline from YAML config.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file.",
    )
    return parser.parse_args()


def main() -> None:
    _assert_gpu_zero_available()
    args = parse_args()
    config_path = Path(args.config)

    orchestrator = ExperimentOrchestrator.from_yaml(config_path)
    orchestrator.run()


if __name__ == "__main__":
    main()
