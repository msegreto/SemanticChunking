from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.factory import DatasetFactory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and prepare raw datasets under data/raw/.")
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name, e.g. 'msmarco-docs' or 'beir/scifact'",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Optional explicit target directory. Defaults to data/raw/<dataset_name>",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Optional dataset split (e.g. train/dev/test). If omitted, processor default is used.",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Only validate local presence; do not download missing files.",
    )
    parser.add_argument(
        "--tasks",
        default="document_retrieval",
        help=(
            "Comma-separated extrinsic tasks to prepare/check. "
            "Supported names: document_retrieval,evidence_retrieval,answer_generation"
        ),
    )
    parser.add_argument(
        "--evidence-path",
        default=None,
        help="Optional local evidence annotations path (used by evidence_retrieval checks).",
    )
    parser.add_argument(
        "--answers-path",
        default=None,
        help="Optional local answer annotations path (used by answer_generation checks).",
    )
    return parser.parse_args()


def _parse_tasks(raw_tasks: str) -> list[str]:
    tasks = []
    seen = set()
    for part in raw_tasks.split(","):
        name = part.strip().lower().replace("-", "_")
        if not name:
            continue
        if name in seen:
            continue
        seen.add(name)
        tasks.append(name)
    return tasks or ["document_retrieval"]


def _check_optional_task_artifacts(
    tasks: list[str],
    evidence_path: str | None,
    answers_path: str | None,
) -> None:
    for task_name in tasks:
        if task_name == "document_retrieval":
            print("[OK] Task 'document_retrieval': base corpus/queries/qrels prepared.")
            continue

        if task_name == "evidence_retrieval":
            if evidence_path and Path(evidence_path).exists():
                print(f"[OK] Task 'evidence_retrieval': evidence file found at {evidence_path}")
            else:
                print(
                    "[WARN] Task 'evidence_retrieval' requested but evidence artifact is missing. "
                    "Skipping this task without failing."
                )
            continue

        if task_name == "answer_generation":
            if answers_path and Path(answers_path).exists():
                print(f"[OK] Task 'answer_generation': answers file found at {answers_path}")
            else:
                print(
                    "[WARN] Task 'answer_generation' requested but answers artifact is missing. "
                    "Skipping this task without failing."
                )
            continue

        print(f"[WARN] Unknown task '{task_name}'. Skipping without failing.")


def main() -> None:
    args = parse_args()
    processor = DatasetFactory.create(args.dataset)

    config = {
        "name": args.dataset,
        "download_if_missing": not args.no_download,
    }
    if args.data_dir:
        config["data_dir"] = args.data_dir
    if args.split:
        config["split"] = args.split

    requested_tasks = _parse_tasks(args.tasks)
    print(f"[INFO] Requested task checks: {requested_tasks}")

    raw_path = processor.ensure_available(config)
    print(f"[OK] Dataset available at: {raw_path}")
    _check_optional_task_artifacts(
        tasks=requested_tasks,
        evidence_path=args.evidence_path,
        answers_path=args.answers_path,
    )


if __name__ == "__main__":
    main()
