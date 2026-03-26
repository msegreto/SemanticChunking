from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
from typing import Any


REQUIRED_UNIT_FIELDS = {
    "parent_docno": str,
    "position": int,
    "text": str,
}


def validate_chunking_config(config: Any, validate_strategy_config: Any) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise TypeError("Chunking config must be a dict.")

    dataset = config.get("dataset")
    if not isinstance(dataset, str) or not dataset.strip():
        raise ValueError("Chunking config requires 'dataset' as a non-empty string.")

    save_chunks = config.get("save_chunks", False)
    if not isinstance(save_chunks, bool):
        raise ValueError("Chunking config requires 'save_chunks' as a boolean.")

    yaml_name = config.get("yaml_name") or config.get("experiment_name") or "default_run"
    if not isinstance(yaml_name, str) or not yaml_name.strip():
        raise ValueError("Chunking config requires a non-empty 'yaml_name' or fallback run name.")

    validated_config = dict(config)
    validated_config["dataset"] = dataset.strip()
    validated_config["save_chunks"] = save_chunks
    validated_config["yaml_name"] = yaml_name.strip()
    return validate_strategy_config(validated_config)


def validate_routed_output(routed_output: Any) -> list[dict[str, Any]]:
    if not isinstance(routed_output, dict):
        raise TypeError("Chunker input must be a dict containing 'split_units'.")

    split_units = routed_output.get("split_units")
    if not isinstance(split_units, list):
        raise TypeError("Chunker input requires 'split_units' as a list.")

    for idx, unit in enumerate(split_units):
        if not isinstance(unit, dict):
            raise TypeError(f"split_units[{idx}] must be a dict.")

        for field_name, expected_type in REQUIRED_UNIT_FIELDS.items():
            if field_name not in unit:
                raise ValueError(f"split_units[{idx}] is missing required field '{field_name}'.")
            if isinstance(unit[field_name], bool) or not isinstance(unit[field_name], expected_type):
                raise TypeError(
                    f"split_units[{idx}]['{field_name}'] must be of type "
                    f"{expected_type.__name__}."
                )

        if not unit["parent_docno"].strip():
            raise ValueError(f"split_units[{idx}]['parent_docno'] must be a non-empty string.")
        if unit["position"] < 0:
            raise ValueError(f"split_units[{idx}]['position'] must be >= 0.")

    return split_units


def group_split_units(split_units: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped_units: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for unit in split_units:
        grouped_units[unit["parent_docno"]].append(unit)
    return grouped_units


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return slug.strip("._") or "item"


def build_run_dir(dataset: str, yaml_name: str) -> Path:
    return Path("data/chunks") / slugify(dataset) / slugify(Path(yaml_name).stem)


def build_doc_path(run_dir: Path, doc_id: str) -> Path:
    safe_doc_id = slugify(doc_id)
    doc_hash = hashlib.sha1(doc_id.encode("utf-8")).hexdigest()[:12]
    return run_dir / f"{safe_doc_id}--{doc_hash}.json"


def load_saved_payload(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None

    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return None

    if not isinstance(payload, dict):
        return None

    chunks = payload.get("chunks")
    if not isinstance(chunks, list):
        return None

    return payload


def build_documents_from_grouped_chunks(
    grouped_units: dict[str, list[dict[str, Any]]],
    grouped_chunks: list[tuple[str, list[dict[str, Any]]]],
    reused: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_chunks: list[dict[str, Any]] = []
    documents: list[dict[str, Any]] = []

    for doc_id, chunks in grouped_chunks:
        all_chunks.extend(chunks)
        documents.append(
            {
                "doc_id": doc_id,
                "num_input_units": len(grouped_units.get(doc_id, [])),
                "num_chunks": len(chunks),
                "reused_existing_chunks": reused,
            }
        )

    return all_chunks, documents


def load_cached_run(
    *,
    run_dir: Path,
    grouped_units: dict[str, list[dict[str, Any]]],
    config: dict[str, Any],
    saved_payload_matches: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]] | None:
    if not run_dir.exists():
        return None

    grouped_chunks: list[tuple[str, list[dict[str, Any]]]] = []

    for doc_id in grouped_units:
        path = build_doc_path(run_dir, doc_id)
        payload = load_saved_payload(path)
        if payload is None:
            return None

        metadata = payload.get("metadata")
        if not saved_payload_matches(metadata=metadata, doc_id=doc_id, config=config):
            return None

        grouped_chunks.append((doc_id, deepcopy(payload["chunks"])))

    if grouped_chunks:
        print(f"[CHUNKING] reusing cached chunks from {run_dir}")

    return build_documents_from_grouped_chunks(grouped_units, grouped_chunks, reused=True)


def save_run_chunks(
    *,
    run_dir: Path,
    grouped_chunks: list[dict[str, Any]],
    config: dict[str, Any],
    build_saved_payload_metadata: Any,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)

    chunks_by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in grouped_chunks:
        chunks_by_doc[chunk["doc_id"]].append(chunk)

    for doc_id, chunks in chunks_by_doc.items():
        path = build_doc_path(run_dir, doc_id)
        payload = {
            "chunks": chunks,
            "metadata": build_saved_payload_metadata(doc_id=doc_id, config=config),
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)


def build_chunk_output(
    *,
    all_chunks: list[dict[str, Any]],
    documents: list[dict[str, Any]],
    config: dict[str, Any],
    run_dir: Path,
    reused_existing_chunks: bool,
    chunking_type: str,
    build_strategy_config_metadata: Any,
) -> dict[str, Any]:
    return {
        "chunks": all_chunks,
        "documents": documents,
        "metadata": {
            "chunking_type": chunking_type,
            "num_documents": len(documents),
            "total_chunks": len(all_chunks),
            "reused_existing_chunks": reused_existing_chunks,
            "chunk_cache_run_dir": str(run_dir),
            "config_used": {
                "dataset": config["dataset"],
                "save_chunks": config["save_chunks"],
                "yaml_name": config["yaml_name"],
                **build_strategy_config_metadata(config),
            },
        },
    }
