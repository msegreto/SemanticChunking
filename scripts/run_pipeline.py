from __future__ import annotations

import argparse
from pathlib import Path

from src.pipelines.experiment_orchestrator import ExperimentOrchestrator


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
    args = parse_args()
    config_path = Path(args.config)

    orchestrator = ExperimentOrchestrator.from_yaml(config_path)
    orchestrator.run()


if __name__ == "__main__":
    main()