from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import yaml

from src.evaluation.extrinsic.factory import ExtrinsicEvaluatorFactory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run extrinsic evaluation.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--index-path", type=str, required=True, help="Path to retrieval index file.")
    parser.add_argument("--index-metadata-path", type=str, required=True, help="Path to retrieval metadata file.")
    parser.add_argument("--index-manifest-path", type=str, required=False, help="Optional retrieval manifest path.")
    return parser.parse_args()


def _resolve_ks(config: dict[str, Any]) -> list[int]:
    evaluation_cfg = config.get("evaluation", {})
    task_cfg = evaluation_cfg.get("extrinsic_tasks", {}).get("document_retrieval", {})
    ks = task_cfg.get("ks", [1, 3, 5, 10])
    return sorted({int(k) for k in ks})


def _resolve_output_path(config: dict[str, Any], config_path: Path) -> Path:
    dataset_name = config.get("dataset", {}).get("name", "unknown-dataset").replace("/", "-")
    chunking_name = config.get("chunking", {}).get("type", "unknown-chunking")

    router_cfg = config.get("router", {})
    routing_name = router_cfg.get("name", "no-routing") if router_cfg.get("enabled", False) else "no-routing"

    experiment_name = (
        config.get("experiment_name")
        or config.get("experiment", {}).get("name")
        or "document-retrieval"
    )

    config_name = config_path.stem.replace("/", "-")

    output_dir = Path(config.get("evaluation", {}).get("results_dir", "results/extrinsic"))
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{dataset_name}_{chunking_name}_{routing_name}_{experiment_name}_{config_name}.csv"
    return output_dir / filename


def _write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    if not rows:
        raise ValueError("No rows to write.")
    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not config.get("evaluation", {}).get("extrinsic", True):
        print("[INFO] Extrinsic evaluation disabled.")
        return

    evaluator_name = config.get("evaluation", {}).get("extrinsic_evaluator", "document_retrieval")
    ks = _resolve_ks(config)

    evaluator = ExtrinsicEvaluatorFactory.create(evaluator_name)
    rows = evaluator.evaluate(
        config=config,
        index_path=Path(args.index_path),
        index_metadata_path=Path(args.index_metadata_path),
        ks=ks,
    )

    output_path = _resolve_output_path(config, config_path)
    _write_csv(rows, output_path)

    print(f"[INFO] Extrinsic evaluation completed. Results saved to: {output_path}")


if __name__ == "__main__":
    main()