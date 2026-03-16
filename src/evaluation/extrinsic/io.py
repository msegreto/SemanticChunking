from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any


def _get_document_retrieval_cfg(config: dict[str, Any]) -> dict[str, Any]:
    return (
        config.get("evaluation", {})
        .get("extrinsic_tasks", {})
        .get("document_retrieval", {})
    )


def check_document_retrieval_prerequisites(config: dict[str, Any]) -> tuple[bool, str]:
    try:
        queries_path = resolve_queries_path(config)
    except Exception as e:
        return False, f"missing queries: {e}"

    try:
        qrels_path = resolve_qrels_path(config)
    except Exception as e:
        return False, f"missing qrels: {e}"

    return True, f"queries={queries_path}, qrels={qrels_path}"


def resolve_queries_path(config: dict[str, Any]) -> Path:
    doc_ret_cfg = _get_document_retrieval_cfg(config)

    normalized_dataset_dir = doc_ret_cfg.get("normalized_dataset_dir")
    if normalized_dataset_dir:
        path = Path(normalized_dataset_dir) / "queries.json"
        if not path.exists():
            raise FileNotFoundError(f"Queries file not found: {path}")
        return path

    dataset_cfg = config.get("dataset", {})
    explicit = dataset_cfg.get("queries_path")
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(f"Queries path does not exist: {path}")
        return path

    data_dir = dataset_cfg.get("data_dir")
    if not data_dir:
        raise ValueError(
            "Could not resolve queries path. Set either "
            "evaluation.extrinsic_tasks.document_retrieval.normalized_dataset_dir "
            "or dataset.queries_path."
        )

    # fallback legacy
    candidates = [
        Path(data_dir) / "queries.json",
        Path(data_dir) / "queries.jsonl",
    ]
    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not resolve queries file from config. "
        "Provide normalized_dataset_dir or dataset.queries_path."
    )


def resolve_qrels_path(config: dict[str, Any]) -> Path:
    doc_ret_cfg = _get_document_retrieval_cfg(config)

    normalized_dataset_dir = doc_ret_cfg.get("normalized_dataset_dir")
    split = doc_ret_cfg.get("split", "dev")

    if normalized_dataset_dir:
        path = Path(normalized_dataset_dir) / "qrels" / f"{split}.json"
        if not path.exists():
            raise FileNotFoundError(f"Qrels file not found: {path}")
        return path

    dataset_cfg = config.get("dataset", {})
    explicit = dataset_cfg.get("qrels_path")
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(f"Qrels path does not exist: {path}")
        return path

    data_dir = dataset_cfg.get("data_dir")
    dataset_split = dataset_cfg.get("split", "dev")
    if not data_dir:
        raise ValueError(
            "Could not resolve qrels path. Set either "
            "evaluation.extrinsic_tasks.document_retrieval.normalized_dataset_dir "
            "or dataset.qrels_path."
        )

    # fallback legacy
    candidates = [
        Path(data_dir) / "qrels" / f"{dataset_split}.json",
        Path(data_dir) / "qrels" / f"{dataset_split}.tsv",
        Path(data_dir) / "qrels.json",
    ]
    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not resolve qrels file from config. "
        "Provide normalized_dataset_dir or dataset.qrels_path."
    )


def load_queries(path: Path) -> dict[str, str]:
    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError(f"queries.json must contain a dict, got: {type(data)}")

        return {str(query_id): str(text) for query_id, text in data.items()}

    if path.suffix == ".jsonl":
        queries: dict[str, str] = {}
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                qid = record.get("_id") or record.get("query_id") or record.get("id")
                text = record.get("text") or record.get("query")
                if qid is None or text is None:
                    continue
                queries[str(qid)] = str(text)
        return queries

    raise ValueError(f"Unsupported queries format: {path}")


def load_qrels(path: Path) -> dict[str, set[str]]:
    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError(f"qrels json must contain a dict, got: {type(data)}")

        qrels: dict[str, set[str]] = {}
        for query_id, doc_map in data.items():
            if not isinstance(doc_map, dict):
                raise ValueError(
                    f"Each qrels entry must be a dict of doc_id -> relevance. "
                    f"Got {type(doc_map)} for query_id={query_id}"
                )
            qrels[str(query_id)] = {
                str(doc_id) for doc_id, rel in doc_map.items() if float(rel) > 0
            }
        return qrels

    if path.suffix == ".tsv":
        import csv

        qrels: dict[str, set[str]] = {}
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                qid = str(row["query-id"])
                doc_id = str(row["corpus-id"])
                score = float(row["score"])
                if score > 0:
                    qrels.setdefault(qid, set()).add(doc_id)
        return qrels

    raise ValueError(f"Unsupported qrels format: {path}")


def load_index_metadata(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        metadata = pickle.load(f)

    if not isinstance(metadata, dict):
        raise ValueError("metadata.pkl must contain a dictionary")
    if "items" not in metadata:
        raise ValueError("metadata.pkl does not contain 'items'")

    return metadata


def load_evidences(path: Path) -> dict[str, list[dict[str, str]]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"evidences json must contain a dict, got: {type(data)}")

    normalized: dict[str, list[dict[str, str]]] = {}
    for query_id, annotations in data.items():
        if not isinstance(annotations, list):
            raise ValueError(
                f"Evidence annotations for query_id={query_id} must be a list, got: {type(annotations)}"
            )

        normalized_annotations: list[dict[str, str]] = []
        for item in annotations:
            if not isinstance(item, dict):
                continue
            doc_id = item.get("doc_id")
            evidence_text = item.get("evidence_text")
            normalized_annotations.append(
                {
                    "doc_id": str(doc_id) if doc_id is not None else "",
                    "evidence_text": str(evidence_text) if evidence_text is not None else "",
                }
            )
        normalized[str(query_id)] = normalized_annotations

    return normalized


def load_answers(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"answers json must contain a dict, got: {type(data)}")

    normalized: dict[str, dict[str, Any]] = {}
    for query_id, payload in data.items():
        if not isinstance(payload, dict):
            raise ValueError(
                f"Answer entry for query_id={query_id} must be a dict, got: {type(payload)}"
            )
        reference_answers = payload.get("reference_answers", [])
        if not isinstance(reference_answers, list):
            reference_answers = []

        normalized[str(query_id)] = {
            "doc_id": str(payload.get("doc_id", "")),
            "answerable": bool(payload.get("answerable", False)),
            "reference_answers": [
                str(answer).strip() for answer in reference_answers if str(answer).strip()
            ],
        }
    return normalized
