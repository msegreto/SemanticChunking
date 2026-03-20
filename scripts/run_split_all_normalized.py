from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.splitting.factory import SplitterFactory


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")
    tmp.replace(path)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _resolve_text(document: dict[str, Any]) -> str:
    for key in ("text", "contents", "body", "document"):
        value = document.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _sanitize_dataset_subpath(dataset_name: str) -> Path:
    parts = [p for p in dataset_name.strip("/").split("/") if p]
    if not parts:
        raise ValueError("Invalid empty dataset name.")
    return Path(*parts)


@dataclass(frozen=True)
class DatasetJob:
    dataset_name: str
    documents_path: Path
    output_path: Path
    state_path: Path
    expected_docs: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run sentence split for all normalized datasets found under a root directory. "
            "Writes per-dataset split output and state_split.json."
        )
    )
    parser.add_argument(
        "--normalized-root",
        type=Path,
        default=Path("data/normalized"),
        help="Root directory containing normalized datasets.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed"),
        help="Root directory where split outputs are written.",
    )
    parser.add_argument(
        "--split-output-name",
        type=str,
        default="sentences.jsonl",
        help="Output split filename for each dataset.",
    )
    parser.add_argument(
        "--state-filename",
        type=str,
        default="state_split.json",
        help="State filename for each dataset.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="en_core_web_sm",
        help="spaCy model used by sentence splitter.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="",
        help=(
            "Optional comma-separated dataset filter by exact dataset_name "
            "(e.g. 'beir/qasper,custom/myset')."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun split even if state already marks dataset as completed.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue processing remaining datasets when one fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print discovered jobs without executing split.",
    )
    return parser.parse_args()


def _parse_dataset_filter(raw: str) -> set[str]:
    out: set[str] = set()
    for item in raw.split(","):
        item = item.strip()
        if item:
            out.add(item)
    return out


def _discover_jobs(
    *,
    normalized_root: Path,
    output_root: Path,
    split_output_name: str,
    state_filename: str,
    dataset_filter: set[str],
) -> list[DatasetJob]:
    jobs: list[DatasetJob] = []

    for metadata_path in sorted(normalized_root.glob("**/metadata.json")):
        meta = _load_json(metadata_path)
        normalized_files = meta.get("normalized_files")
        if not isinstance(normalized_files, dict):
            continue
        docs_rel = normalized_files.get("documents")
        if not isinstance(docs_rel, str) or not docs_rel:
            continue

        dataset_name = meta.get("dataset_name")
        if not isinstance(dataset_name, str) or not dataset_name.strip():
            rel_dir = metadata_path.parent.relative_to(normalized_root)
            dataset_name = str(rel_dir).replace("\\", "/")
        dataset_name = dataset_name.strip()

        if dataset_filter and dataset_name not in dataset_filter:
            continue

        documents_path = metadata_path.parent / docs_rel
        if not documents_path.exists():
            continue

        expected_docs = meta.get("num_documents")
        if not isinstance(expected_docs, int) or expected_docs < 0:
            expected_docs = -1

        dataset_subpath = _sanitize_dataset_subpath(dataset_name)
        dataset_output_dir = output_root / dataset_subpath
        jobs.append(
            DatasetJob(
                dataset_name=dataset_name,
                documents_path=documents_path.resolve(),
                output_path=(dataset_output_dir / split_output_name).resolve(),
                state_path=(dataset_output_dir / state_filename).resolve(),
                expected_docs=expected_docs,
            )
        )

    return jobs


def _run_one_dataset(job: DatasetJob, *, model_name: str, force: bool) -> tuple[bool, str]:
    old_state = _load_json(job.state_path)
    if not force and bool(old_state.get("completed", False)):
        return True, "already completed, skipped"

    splitter = SplitterFactory.create("sentence")
    split_cfg = {"model": model_name}
    components = splitter.build_streaming_components(split_cfg)

    job.output_path.parent.mkdir(parents=True, exist_ok=True)
    job.output_path.write_text("", encoding="utf-8")

    started_at = datetime.now(timezone.utc)
    started_iso = started_at.replace(microsecond=0).isoformat()
    running_state = {
        "version": 1,
        "status": "running",
        "completed": False,
        "created_at": old_state.get("created_at", started_iso),
        "updated_at": started_iso,
        "started_at": started_iso,
        "finished_at": None,
        "duration_seconds": None,
        "dataset_name": job.dataset_name,
        "split_type": "sentence",
        "spacy_model": model_name,
        "split_output_path": str(job.output_path),
        "normalized_documents_path": str(job.documents_path),
        "expected_docs": job.expected_docs,
        "num_docs_processed": 0,
        "num_units_processed": 0,
        "next_unit_id": 0,
        "error": None,
    }
    _save_json(job.state_path, running_state)

    docs_processed = 0
    units_processed = 0
    next_unit_id = 0

    try:
        with job.documents_path.open("r", encoding="utf-8") as f_in, job.output_path.open(
            "a", encoding="utf-8"
        ) as f_out:
            for raw_line in f_in:
                line = raw_line.strip()
                if not line:
                    continue
                document = json.loads(line)
                doc_id = str(document.get("doc_id") or document.get("id") or "").strip()
                if not doc_id:
                    raise ValueError("Found document without doc_id/id in normalized documents file.")

                text = _resolve_text(document)
                units, next_unit_id = splitter.split_document_streaming(
                    doc_id=doc_id,
                    text=text,
                    unit_id_start=next_unit_id,
                    nlp=components["nlp"],
                    max_chars_per_batch=int(components["max_chars_per_batch"]),
                )
                for unit in units:
                    f_out.write(json.dumps(unit, ensure_ascii=False) + "\n")

                docs_processed += 1
                units_processed += len(units)

        finished_at = datetime.now(timezone.utc)
        finished_iso = finished_at.replace(microsecond=0).isoformat()
        duration = (finished_at - started_at).total_seconds()
        final_state = dict(running_state)
        final_state.update(
            {
                "status": "completed",
                "completed": True,
                "updated_at": finished_iso,
                "finished_at": finished_iso,
                "duration_seconds": duration,
                "num_docs_processed": docs_processed,
                "num_units_processed": units_processed,
                "next_unit_id": next_unit_id,
                "error": None,
            }
        )
        _save_json(job.state_path, final_state)
        return True, f"completed docs={docs_processed} units={units_processed}"
    except Exception as e:
        failed_at = _utc_now_iso()
        fail_state = dict(running_state)
        fail_state.update(
            {
                "status": "failed",
                "completed": False,
                "updated_at": failed_at,
                "finished_at": failed_at,
                "num_docs_processed": docs_processed,
                "num_units_processed": units_processed,
                "next_unit_id": next_unit_id,
                "error": str(e),
            }
        )
        _save_json(job.state_path, fail_state)
        return False, str(e)


def main() -> None:
    args = parse_args()
    normalized_root = args.normalized_root
    output_root = args.output_root

    if not normalized_root.exists():
        raise FileNotFoundError(f"Normalized root not found: {normalized_root}")

    dataset_filter = _parse_dataset_filter(args.datasets)
    jobs = _discover_jobs(
        normalized_root=normalized_root,
        output_root=output_root,
        split_output_name=args.split_output_name,
        state_filename=args.state_filename,
        dataset_filter=dataset_filter,
    )
    if not jobs:
        print("[WARN] No normalized datasets found to process.")
        return

    print(f"[INFO] Found {len(jobs)} normalized dataset(s).")
    for job in jobs:
        print(
            f"[INFO] job dataset={job.dataset_name} docs={job.documents_path} "
            f"output={job.output_path} state={job.state_path}"
        )

    if args.dry_run:
        print("[DRY-RUN] Split not executed.")
        return

    failed = 0
    for idx, job in enumerate(jobs, start=1):
        print(f"[INFO] ({idx}/{len(jobs)}) Processing {job.dataset_name}")
        ok, details = _run_one_dataset(job, model_name=args.model, force=args.force)
        if ok:
            print(f"[OK] {job.dataset_name}: {details}")
        else:
            failed += 1
            print(f"[FAIL] {job.dataset_name}: {details}")
            if not args.continue_on_error:
                break

    if failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
