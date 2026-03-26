from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.loader import load_experiment_config
from src.datasets.service import process_dataset


@dataclass(frozen=True)
class DatasetJobKey:
    dataset_name: str
    split: str
    normalized_dir: str


@dataclass
class DatasetJob:
    key: DatasetJobKey
    config: dict[str, Any]
    source_configs: list[Path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pre-download and normalize all datasets required by experiment YAML files. "
            "This fills data/normalized/ ahead of full runs."
        )
    )
    parser.add_argument(
        "--configs-dir",
        type=Path,
        default=Path("configs/experiments/grid_search_stage1_stella"),
        help="Directory containing experiment YAML files.",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="*.yaml",
        help="File glob to select experiment YAML files inside --configs-dir.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="",
        help=(
            "Optional comma-separated dataset filter (e.g. 'beir/qasper,beir/techqa'). "
            "When omitted, all datasets found in YAML files are prepared."
        ),
    )
    parser.add_argument(
        "--normalized-root",
        type=Path,
        default=None,
        help=(
            "Optional root override for normalized output. "
            "Example: --normalized-root normalized -> normalized/beir/qasper"
        ),
    )
    parser.add_argument(
        "--split-override",
        type=str,
        default=None,
        help="Optional split override applied to all dataset jobs (e.g. test/dev).",
    )
    parser.add_argument(
        "--force-rebuild-normalized",
        action="store_true",
        help="Ignore existing normalized cache and rebuild from raw/source.",
    )
    parser.add_argument(
        "--keep-raw",
        action="store_true",
        help="Do not delete data/raw after writing normalized cache.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue preparing remaining datasets even if one fails.",
    )
    parser.add_argument(
        "--skip-unavailable",
        action="store_true",
        default=True,
        help=(
            "Skip datasets that fail because they are not publicly downloadable "
            "(default: enabled)."
        ),
    )
    parser.add_argument(
        "--no-skip-unavailable",
        dest="skip_unavailable",
        action="store_false",
        help="Treat non-public/unavailable datasets as hard failures.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print planned dataset jobs without running downloads/conversion.",
    )
    return parser.parse_args()


def _normalize_task_name(task_name: Any) -> str | None:
    if not isinstance(task_name, str):
        return None
    normalized = task_name.strip().lower().replace("-", "_")
    return normalized or None


def _parse_requested_tasks(config: dict[str, Any]) -> list[str]:
    evaluation_cfg = config.get("evaluation", {})
    requested = []
    if isinstance(evaluation_cfg, dict):
        raw_tasks = evaluation_cfg.get("extrinsic_tasks_to_run")
        if isinstance(raw_tasks, list):
            requested = raw_tasks
        elif isinstance(evaluation_cfg.get("extrinsic_evaluator"), str):
            requested = [evaluation_cfg["extrinsic_evaluator"]]

    out: list[str] = []
    seen: set[str] = set()
    for task in requested:
        normalized = _normalize_task_name(task)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized)

    if "document_retrieval" not in seen:
        out.insert(0, "document_retrieval")

    return out


def _parse_dataset_filter(raw: str) -> set[str]:
    out: set[str] = set()
    for item in raw.split(","):
        name = item.strip()
        if name:
            out.add(name)
    return out


def _normalize_subpath(dataset_name: str) -> Path:
    return Path(*[p for p in dataset_name.strip("/").split("/") if p])


def _resolve_normalized_dir(
    *,
    dataset_cfg: dict[str, Any],
    dataset_name: str,
    normalized_root: Path | None,
) -> Path:
    if normalized_root is not None:
        return (normalized_root / _normalize_subpath(dataset_name)).expanduser().resolve()

    explicit = dataset_cfg.get("normalized_dir")
    if isinstance(explicit, str) and explicit.strip():
        return Path(explicit).expanduser().resolve()

    return (Path("data/normalized") / _normalize_subpath(dataset_name)).resolve()


def discover_jobs(
    *,
    configs_dir: Path,
    glob_expr: str,
    dataset_filter: set[str],
    normalized_root: Path | None,
    split_override: str | None,
    force_rebuild_normalized: bool,
    keep_raw: bool,
) -> list[DatasetJob]:
    files = sorted(p for p in configs_dir.glob(glob_expr) if p.is_file())
    if not files:
        return []

    grouped: dict[DatasetJobKey, DatasetJob] = {}

    for config_path in files:
        cfg = load_experiment_config(config_path)
        dataset_cfg = cfg.get("dataset")
        if not isinstance(dataset_cfg, dict):
            continue

        dataset_name = str(dataset_cfg.get("name", "")).strip()
        if not dataset_name:
            continue

        if dataset_filter and dataset_name not in dataset_filter:
            continue

        split = str(split_override or dataset_cfg.get("split", "test"))
        normalized_dir = _resolve_normalized_dir(
            dataset_cfg=dataset_cfg,
            dataset_name=dataset_name,
            normalized_root=normalized_root,
        )

        key = DatasetJobKey(
            dataset_name=dataset_name,
            split=split,
            normalized_dir=str(normalized_dir),
        )

        tasks = _parse_requested_tasks(cfg)

        if key not in grouped:
            merged_cfg = dict(dataset_cfg)
            merged_cfg["name"] = dataset_name
            merged_cfg["split"] = split
            merged_cfg["normalized_dir"] = str(normalized_dir)
            merged_cfg["required_extrinsic_tasks"] = tasks
            merged_cfg["download_if_missing"] = True
            if force_rebuild_normalized:
                merged_cfg["force_rebuild_normalized"] = True
                merged_cfg["use_normalized_if_available"] = False
            if keep_raw:
                merged_cfg["delete_raw_after_normalized"] = False

            grouped[key] = DatasetJob(
                key=key,
                config=merged_cfg,
                source_configs=[config_path.resolve()],
            )
            continue

        job = grouped[key]
        source = job.config.get("required_extrinsic_tasks", ["document_retrieval"])
        if not isinstance(source, list):
            source = ["document_retrieval"]
        merged_tasks = list(source)
        for task in tasks:
            if task not in merged_tasks:
                merged_tasks.append(task)
        job.config["required_extrinsic_tasks"] = merged_tasks
        job.source_configs.append(config_path.resolve())

    return sorted(grouped.values(), key=lambda j: (j.key.dataset_name, j.key.split, j.key.normalized_dir))


def main() -> None:
    args = parse_args()
    configs_dir = args.configs_dir
    if not configs_dir.exists():
        raise FileNotFoundError(f"Config directory not found: {configs_dir}")

    dataset_filter = _parse_dataset_filter(args.datasets)

    jobs = discover_jobs(
        configs_dir=configs_dir,
        glob_expr=args.glob,
        dataset_filter=dataset_filter,
        normalized_root=args.normalized_root,
        split_override=args.split_override,
        force_rebuild_normalized=args.force_rebuild_normalized,
        keep_raw=args.keep_raw,
    )

    if not jobs:
        print("[WARN] No dataset jobs found from the selected configs/filter.")
        return

    print(f"[INFO] Dataset jobs to prepare: {len(jobs)}")
    for idx, job in enumerate(jobs, start=1):
        print(
            f"[PLAN] ({idx}/{len(jobs)}) dataset={job.key.dataset_name} split={job.key.split} "
            f"normalized_dir={job.key.normalized_dir} "
            f"tasks={job.config.get('required_extrinsic_tasks', ['document_retrieval'])} "
            f"configs={len(job.source_configs)}"
        )

    if args.dry_run:
        return

    failures: list[tuple[DatasetJob, Exception]] = []
    skipped_unavailable: list[tuple[DatasetJob, Exception]] = []
    for idx, job in enumerate(jobs, start=1):
        print(
            f"[INFO] ({idx}/{len(jobs)}) Preparing dataset '{job.key.dataset_name}' "
            f"(split={job.key.split})..."
        )
        try:
            payload = process_dataset(job.config, force_rebuild=bool(job.config.get("force_rebuild_normalized", False)))
            metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
            loaded_from = metadata.get("loaded_from", "unknown")
            num_documents = metadata.get("num_documents", "n/a")
            num_queries = metadata.get("num_queries", "n/a")
            print(
                f"[OK] dataset={job.key.dataset_name} split={job.key.split} "
                f"loaded_from={loaded_from} docs={num_documents} queries={num_queries} "
                f"normalized_dir={job.key.normalized_dir}"
            )
        except Exception as exc:  # pragma: no cover
            message = str(exc).lower()
            if args.skip_unavailable and "not publicly downloadable" in message:
                skipped_unavailable.append((job, exc))
                print(
                    f"[WARN] dataset={job.key.dataset_name} split={job.key.split} skipped: {exc}"
                )
                continue
            failures.append((job, exc))
            print(
                f"[ERROR] dataset={job.key.dataset_name} split={job.key.split} failed: {exc}"
            )
            if not args.continue_on_error:
                break

    if skipped_unavailable:
        print("[SUMMARY] Skipped unavailable datasets:")
        for job, exc in skipped_unavailable:
            print(
                f"[SKIPPED] dataset={job.key.dataset_name} split={job.key.split} "
                f"normalized_dir={job.key.normalized_dir} reason={exc}"
            )

    if failures:
        print("[SUMMARY] Some dataset jobs failed.")
        for job, exc in failures:
            print(
                f"[FAILED] dataset={job.key.dataset_name} split={job.key.split} "
                f"normalized_dir={job.key.normalized_dir} error={exc}"
            )
        raise SystemExit(1)

    print("[SUMMARY] All dataset jobs completed successfully.")
    manifest = [
        {
            "dataset": job.key.dataset_name,
            "split": job.key.split,
            "normalized_dir": job.key.normalized_dir,
            "required_extrinsic_tasks": job.config.get("required_extrinsic_tasks", ["document_retrieval"]),
            "source_configs": [str(p) for p in job.source_configs],
        }
        for job in jobs
    ]
    print("[SUMMARY] Prepared manifest:")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
