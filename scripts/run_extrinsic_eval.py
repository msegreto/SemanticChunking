from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from src.config.loader import load_experiment_config
from src.evaluation.extrinsic.common import build_base_row, build_run_context
from src.evaluation.extrinsic.factory import ExtrinsicEvaluatorFactory
from src.evaluation.extrinsic.io import (
    check_document_retrieval_prerequisites,
    resolve_answers_path,
    resolve_evidence_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run extrinsic evaluation.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--index-path", type=str, required=True, help="Path to retrieval index file.")
    parser.add_argument("--index-metadata-path", type=str, required=True, help="Path to retrieval metadata file.")
    parser.add_argument("--index-manifest-path", type=str, required=False, help="Optional retrieval manifest path.")
    return parser.parse_args()


def _normalize_task_name(task_name: str) -> str:
    return task_name.strip().lower().replace("-", "_")


def _resolve_requested_tasks(config: dict[str, Any]) -> list[str]:
    evaluation_cfg = config.get("evaluation", {})
    requested_tasks = evaluation_cfg.get("extrinsic_tasks_to_run")

    if requested_tasks is None:
        fallback_task = evaluation_cfg.get("extrinsic_evaluator", "document_retrieval")
        requested_tasks = [fallback_task]

    if not isinstance(requested_tasks, list) or not requested_tasks:
        raise ValueError(
            "'evaluation.extrinsic_tasks_to_run' must be a non-empty list of task names."
        )

    normalized: list[str] = []
    seen: set[str] = set()
    for task_name in requested_tasks:
        if not isinstance(task_name, str) or not task_name.strip():
            raise ValueError(
                "Each entry in 'evaluation.extrinsic_tasks_to_run' must be a non-empty string."
            )
        name = _normalize_task_name(task_name)
        if name not in seen:
            seen.add(name)
            normalized.append(name)
    return normalized


def _resolve_ks_for_task(config: dict[str, Any], task_name: str) -> list[int]:
    evaluation_cfg = config.get("evaluation", {})
    task_cfg = evaluation_cfg.get("extrinsic_tasks", {}).get(task_name, {})
    ks = task_cfg.get("ks", [1, 3, 5, 10])

    if not isinstance(ks, list) or not ks:
        raise ValueError(
            f"'evaluation.extrinsic_tasks.{task_name}.ks' must be a non-empty list."
        )

    resolved = sorted({int(k) for k in ks})
    if any(k <= 0 for k in resolved):
        raise ValueError(
            f"'evaluation.extrinsic_tasks.{task_name}.ks' must contain only positive integers."
        )
    return resolved


def _check_required_task_file(
    *,
    task_name: str,
    config: dict[str, Any],
) -> tuple[bool, str]:
    if task_name == "evidence_retrieval":
        label = "evidence"
        resolver = resolve_evidence_path
        key = "evidence_path"
    elif task_name == "answer_generation":
        label = "answers"
        resolver = resolve_answers_path
        key = "answers_path"
    else:
        return False, f"unsupported task for required file check: {task_name}"

    try:
        value = resolver(config)
    except Exception as e:
        return False, f"missing {key}: {e}"

    if not value.exists():
        return False, f"{label} file does not exist: {value}"
    return True, f"{key}={value}"


def _task_support_status(config: dict[str, Any], task_name: str) -> tuple[bool, str]:
    if task_name == "document_retrieval":
        return check_document_retrieval_prerequisites(config)

    if task_name == "evidence_retrieval":
        return _check_required_task_file(task_name=task_name, config=config)
    if task_name == "answer_generation":
        return _check_required_task_file(task_name=task_name, config=config)

    return True, "no pre-check configured"


def _build_skipped_rows(
    *,
    config: dict[str, Any],
    task_name: str,
    ks: list[int],
    details: str,
    index_path: Path,
    index_metadata_path: Path,
) -> list[dict[str, Any]]:
    context = build_run_context(
        config=config,
        task_name=task_name,
        default_experiment_name=task_name,
    )
    rows: list[dict[str, Any]] = []
    for k in ks:
        row = build_base_row(
            context=context,
            k=int(k),
            index_path=index_path,
            index_metadata_path=index_metadata_path,
        )
        row.update(
            {
                "status": "skipped",
                "details": details,
            }
        )
        rows.append(row)
    return rows


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

    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key in seen:
                continue
            seen.add(key)
            fieldnames.append(key)

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def main() -> None:
    args = parse_args()

    config_path = Path(args.config)
    config = load_experiment_config(config_path)

    if not config.get("evaluation", {}).get("extrinsic", True):
        print("[INFO] Extrinsic evaluation disabled.")
        return

    requested_tasks = _resolve_requested_tasks(config)
    rows: list[dict[str, Any]] = []

    for task_name in requested_tasks:
        ks = _resolve_ks_for_task(config, task_name)
        supported, support_reason = _task_support_status(config, task_name)

        if not supported:
            print(f"[WARN] Skipping extrinsic task '{task_name}': {support_reason}")
            rows.extend(
                _build_skipped_rows(
                    config=config,
                    task_name=task_name,
                    ks=ks,
                    details=f"task not supported for current config: {support_reason}",
                    index_path=Path(args.index_path),
                    index_metadata_path=Path(args.index_metadata_path),
                )
            )
            continue

        evaluator = ExtrinsicEvaluatorFactory.create(task_name)
        print(f"[INFO] Running extrinsic task: {task_name} with ks={ks}")
        try:
            task_rows = evaluator.evaluate(
                config=config,
                index_path=Path(args.index_path),
                index_metadata_path=Path(args.index_metadata_path),
                ks=ks,
            )
            rows.extend(task_rows)
        except Exception as e:
            print(f"[WARN] Task '{task_name}' failed, recording as skipped: {e}")
            rows.extend(
                _build_skipped_rows(
                    config=config,
                    task_name=task_name,
                    ks=ks,
                    details=f"task failed at runtime: {e}",
                    index_path=Path(args.index_path),
                    index_metadata_path=Path(args.index_metadata_path),
                )
            )

    output_path = _resolve_output_path(config, config_path)
    _write_csv(rows, output_path)

    print(f"[INFO] Extrinsic evaluation completed. Results saved to: {output_path}")


if __name__ == "__main__":
    main()
