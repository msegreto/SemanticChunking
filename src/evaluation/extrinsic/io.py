from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd

from src.datasets.service import get_split


_TASK_ALIASES: dict[str, str] = {
    "document_retrieval": "document_retrieval",
    "doc_retrieval": "document_retrieval",
    "document": "document_retrieval",
    "evidence_retrieval": "evidence_retrieval",
    "evidence": "evidence_retrieval",
    "answer_generation": "answer_generation",
    "generation": "answer_generation",
    "qa_generation": "answer_generation",
}


class JsonlItemStore:
    def __init__(self, jsonl_path: Path, *, expected_count: int | None = None) -> None:
        self.path = Path(jsonl_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Items JSONL not found: {self.path}")
        self._offsets = self._load_or_build_offsets()
        if expected_count is not None and len(self._offsets) != int(expected_count):
            raise ValueError(
                f"Items JSONL count mismatch: expected {expected_count}, found {len(self._offsets)} at {self.path}"
            )

    def _index_path(self) -> Path:
        return self.path.with_suffix(self.path.suffix + ".idx")

    def _load_or_build_offsets(self) -> list[int]:
        index_path = self._index_path()
        if index_path.exists():
            try:
                raw = json.loads(index_path.read_text(encoding="utf-8"))
                if isinstance(raw, list) and all(isinstance(x, int) for x in raw):
                    return raw
            except Exception:
                pass

        offsets: list[int] = []
        with self.path.open("rb") as f:
            while True:
                pos = f.tell()
                line = f.readline()
                if not line:
                    break
                if line.strip():
                    offsets.append(pos)

        try:
            index_path.write_text(json.dumps(offsets), encoding="utf-8")
        except Exception:
            # Non-fatal: offset cache can be rebuilt.
            pass
        return offsets

    def __len__(self) -> int:
        return len(self._offsets)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if idx < 0:
            idx = len(self._offsets) + idx
        if idx < 0 or idx >= len(self._offsets):
            raise IndexError(idx)
        with self.path.open("rb") as f:
            f.seek(self._offsets[idx])
            line = f.readline()
        if not line:
            raise IndexError(idx)
        return json.loads(line.decode("utf-8"))


def _get_document_retrieval_cfg(config: dict[str, Any]) -> dict[str, Any]:
    return get_extrinsic_task_cfg(config, "document_retrieval")


def canonicalize_task_name(task_name: str) -> str:
    normalized = task_name.strip().lower().replace("-", "_")
    return _TASK_ALIASES.get(normalized, normalized)


def get_extrinsic_task_cfg(config: dict[str, Any], task_name: str) -> dict[str, Any]:
    normalized = canonicalize_task_name(task_name)
    extrinsic_tasks = config.get("evaluation", {}).get("extrinsic_tasks", {})
    if not isinstance(extrinsic_tasks, dict):
        return {}

    candidates = [normalized]
    for alias, canonical in _TASK_ALIASES.items():
        if canonical == normalized and alias != normalized:
            candidates.append(alias)

    for candidate in candidates:
        value = extrinsic_tasks.get(candidate)
        if isinstance(value, dict):
            return value
    return {}


def resolve_normalized_dataset_dir(config: dict[str, Any]) -> Path | None:
    doc_ret_cfg = _get_document_retrieval_cfg(config)
    normalized_dataset_dir = doc_ret_cfg.get("normalized_dataset_dir")
    if isinstance(normalized_dataset_dir, str) and normalized_dataset_dir.strip():
        return Path(normalized_dataset_dir)

    dataset_cfg = config.get("dataset", {})
    dataset_normalized_dir = dataset_cfg.get("normalized_dir")
    if isinstance(dataset_normalized_dir, str) and dataset_normalized_dir.strip():
        return Path(dataset_normalized_dir)

    dataset_name = dataset_cfg.get("name")
    if isinstance(dataset_name, str) and dataset_name.strip():
        return Path("data") / "normalized" / dataset_name
    return None


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
    normalized_dataset_dir = resolve_normalized_dataset_dir(config)
    if normalized_dataset_dir is not None:
        path = normalized_dataset_dir / "topics.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Topics file not found: {path}")
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

    path = Path(data_dir) / "topics.jsonl"
    if path.exists():
        return path

    raise FileNotFoundError(
        "Could not resolve queries file from config. "
        "Provide normalized_dataset_dir or dataset.queries_path."
    )


def resolve_qrels_path(config: dict[str, Any]) -> Path:
    doc_ret_cfg = _get_document_retrieval_cfg(config)
    dataset_cfg = config.get("dataset", {})

    normalized_dataset_dir = resolve_normalized_dataset_dir(config)
    split_override = doc_ret_cfg.get("split")
    if isinstance(split_override, str) and split_override.strip():
        split = split_override.strip()
    else:
        split = get_split(dataset_cfg)

    if normalized_dataset_dir is not None:
        path = normalized_dataset_dir / "qrels" / f"{split}.tsv"
        if not path.exists():
            raise FileNotFoundError(f"Qrels file not found: {path}")
        return path

    explicit = dataset_cfg.get("qrels_path")
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(f"Qrels path does not exist: {path}")
        return path

    data_dir = dataset_cfg.get("data_dir")
    if not data_dir:
        raise ValueError(
            "Could not resolve qrels path. Set either "
            "evaluation.extrinsic_tasks.document_retrieval.normalized_dataset_dir "
            "or dataset.qrels_path."
        )

    path = Path(data_dir) / "qrels" / f"{split}.tsv"
    if path.exists():
        return path

    raise FileNotFoundError(
        "Could not resolve qrels file from config. "
        "Provide normalized_dataset_dir or dataset.qrels_path."
    )


def resolve_evidence_path(config: dict[str, Any]) -> Path:
    task_cfg = get_extrinsic_task_cfg(config, "evidence_retrieval")
    evidence_path = task_cfg.get("evidence_path")
    if isinstance(evidence_path, str) and evidence_path.strip():
        return Path(evidence_path)

    normalized_dataset_dir = resolve_normalized_dataset_dir(config)
    if normalized_dataset_dir is not None:
        return normalized_dataset_dir / "evidences.json"
    raise ValueError("Could not resolve evidence path from config.")


def resolve_answers_path(config: dict[str, Any]) -> Path:
    task_cfg = get_extrinsic_task_cfg(config, "answer_generation")
    answers_path = task_cfg.get("answers_path")
    if isinstance(answers_path, str) and answers_path.strip():
        return Path(answers_path)

    normalized_dataset_dir = resolve_normalized_dataset_dir(config)
    if normalized_dataset_dir is not None:
        return normalized_dataset_dir / "answers.json"
    raise ValueError("Could not resolve answers path from config.")


def load_queries(path: Path) -> dict[str, str]:
    if path.suffix == ".jsonl":
        queries: dict[str, str] = {}
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                qid = record.get("qid")
                text = record.get("query")
                if qid is None or text is None:
                    continue
                queries[str(qid)] = str(text)
        return queries

    raise ValueError(f"Unsupported queries format: {path}")


def load_qrels(path: Path) -> dict[str, dict[str, float]]:
    if path.suffix == ".tsv":
        import csv

        qrels: dict[str, dict[str, float]] = {}
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                qid = str(row.get("qid") or "")
                doc_id = str(row.get("docno") or "")
                score = float(row.get("label") or 0.0)
                if not qid or not doc_id:
                    continue
                if score > 0:
                    qrels.setdefault(qid, {})[doc_id] = score
        return qrels

    raise ValueError(f"Unsupported qrels format: {path}")


def load_index_metadata(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        metadata = pickle.load(f)

    if not isinstance(metadata, dict):
        raise ValueError("metadata.pkl must contain a dictionary")
    if "items" not in metadata:
        jsonl_path = metadata.get("items_jsonl_path")
        if isinstance(jsonl_path, str) and jsonl_path.strip():
            metadata["items"] = JsonlItemStore(
                Path(jsonl_path),
                expected_count=metadata.get("num_items"),
            )
        else:
            raise ValueError("metadata.pkl does not contain 'items' or 'items_jsonl_path'")

    return metadata


def load_run_tsv(path: Path) -> dict[str, list[dict[str, Any]]]:
    df = pd.read_csv(path, sep="\t")
    required = {"qid", "docno", "score"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Run TSV is missing required columns: {sorted(missing)}")

    run: dict[str, list[dict[str, Any]]] = {}
    if "rank" in df.columns:
        df = df.sort_values(["qid", "rank", "score"], ascending=[True, True, False])
    else:
        df = df.sort_values(["qid", "score"], ascending=[True, False])

    for row in df.to_dict(orient="records"):
        qid = str(row["qid"])
        run.setdefault(qid, []).append(
            {
                "docno": str(row["docno"]),
                "score": float(row["score"]),
            }
        )
    return run


def load_items_by_docno(path: Path) -> dict[str, dict[str, Any]]:
    items: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            docno = str(row.get("docno") or "").strip()
            if not docno:
                continue
            items[docno] = row
    return items


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
