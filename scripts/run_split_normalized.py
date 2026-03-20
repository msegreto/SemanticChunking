from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config.loader import load_experiment_config
from src.datasets.factory import DatasetFactory
from src.splitting.factory import SplitterFactory


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run only the split stage starting from an already normalized dataset "
            "and persist completion state to JSON."
        )
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to experiment YAML config.")
    parser.add_argument(
        "--state-file",
        type=Path,
        default=None,
        help=(
            "Optional path for split state JSON. "
            "Default: <split_output_dir>/state_split.json"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rerun even if state file already marks the split as completed.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show resolved paths and planned work without running the split.",
    )
    return parser.parse_args()


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")
    tmp.replace(path)


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _resolve_state_path(explicit: Path | None, split_output_path: Path) -> Path:
    if explicit is not None:
        return explicit
    return split_output_path.parent / "state_split.json"


def _resolve_text(document: dict[str, Any]) -> str:
    for key in ("text", "contents", "body", "document"):
        value = document.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def main() -> None:
    args = parse_args()
    config_path = args.config
    config = load_experiment_config(config_path)

    dataset_cfg = dict(config.get("dataset", {}))
    dataset_name = str(dataset_cfg.get("name", "")).strip()
    if not dataset_name:
        raise ValueError("Missing required config: dataset.name")

    split_cfg = dict(config.get("split", {}))
    split_type = str(split_cfg.get("type", "")).strip()
    if not split_type:
        raise ValueError("Missing required config: split.type")
    if split_type != "sentence":
        raise ValueError(
            "This script currently supports only split.type='sentence' for streaming split."
        )

    splitter = SplitterFactory.create(split_type)
    split_output_path = splitter.resolve_output_path(split_cfg)
    if split_output_path is None:
        raise ValueError(
            "Missing required config: split.output_path. "
            "Set split.save_output=true and split.output_path in YAML."
        )

    state_path = _resolve_state_path(args.state_file, split_output_path)
    old_state = _load_state(state_path)
    if not args.force and bool(old_state.get("completed", False)):
        print(f"[INFO] Split already completed according to state file: {state_path}")
        print("[INFO] Nothing to do. Use --force to rerun.")
        return

    processor = DatasetFactory.create(dataset_name)
    normalized_data = processor.try_load_reusable_normalized(dataset_cfg)
    if normalized_data is None:
        raise RuntimeError(
            "Normalized dataset not available/reusable for this config. "
            "Prepare normalized data first, then rerun this script."
        )

    metadata = normalized_data.get("metadata", {})
    normalized_path = metadata.get("normalized_path")
    normalized_files = metadata.get("normalized_files", {})
    if not isinstance(normalized_path, str) or not normalized_path:
        raise ValueError("Dataset metadata is missing normalized_path.")
    if not isinstance(normalized_files, dict):
        raise ValueError("Dataset metadata is missing normalized_files.")
    docs_rel = normalized_files.get("documents")
    if not isinstance(docs_rel, str) or not docs_rel:
        raise ValueError("Dataset metadata is missing normalized documents filename.")

    documents_path = Path(normalized_path) / docs_rel
    if not documents_path.exists():
        raise FileNotFoundError(f"Normalized documents file not found: {documents_path}")

    expected_docs = metadata.get("num_documents")
    if not isinstance(expected_docs, int) or expected_docs < 0:
        expected_docs = len(normalized_data.get("documents", {}))

    print(f"[INFO] Config: {config_path}")
    print(f"[INFO] Dataset: {dataset_name}")
    print(f"[INFO] Normalized documents: {documents_path}")
    print(f"[INFO] Split output: {split_output_path}")
    print(f"[INFO] State file: {state_path}")
    print(f"[INFO] Expected docs: {expected_docs}")

    if args.dry_run:
        print("[DRY-RUN] Split not executed.")
        return

    split_output_path.parent.mkdir(parents=True, exist_ok=True)
    split_output_path.write_text("", encoding="utf-8")

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
        "config_path": str(config_path.resolve()),
        "dataset_name": dataset_name,
        "split_type": split_type,
        "split_output_path": str(split_output_path.resolve()),
        "normalized_documents_path": str(documents_path.resolve()),
        "expected_docs": int(expected_docs),
        "num_docs_processed": 0,
        "num_units_processed": 0,
        "next_unit_id": 0,
        "error": None,
    }
    _save_json(state_path, running_state)

    next_unit_id = 0
    docs_processed = 0
    units_processed = 0
    components = splitter.build_streaming_components(split_cfg)

    try:
        with documents_path.open("r", encoding="utf-8") as f_in, split_output_path.open(
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
        _save_json(state_path, final_state)

        print(
            "[INFO] Split completed successfully: "
            f"docs={docs_processed}, units={units_processed}, output={split_output_path}"
        )
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
        _save_json(state_path, fail_state)
        raise


if __name__ == "__main__":
    main()
