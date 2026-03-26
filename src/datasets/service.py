from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.datasets.pt_dataset import PyTerrierNormalizedDataset, pt


NORMALIZED_SCHEMA_VERSION = "2.0"


@dataclass(frozen=True)
class DatasetSpec:
    kind: str
    default_split: str
    script_name: str | None = None
    pyterrier_name: str | None = None


_SPECS: dict[str, DatasetSpec] = {
    "beir/qasper": DatasetSpec(kind="script", default_split="test", script_name="download_qasper_hf_to_normalized.py"),
    "beir/techqa": DatasetSpec(kind="script", default_split="test", script_name="download_techqa_hf_to_normalized.py"),
    "msmarco": DatasetSpec(kind="pyterrier", default_split="dev", pyterrier_name="irds:msmarco-document"),
    "msmarco-docs": DatasetSpec(kind="pyterrier", default_split="dev", pyterrier_name="irds:msmarco-document"),
}


def get_dataset(config: dict[str, Any]) -> PyTerrierNormalizedDataset:
    dataset_name = str(config.get("name", "")).strip()
    split = get_split(config)
    normalized_dir = resolve_normalized_dir(config)
    return PyTerrierNormalizedDataset(dataset_name=dataset_name, normalized_dir=normalized_dir, split=split)


def get_split(config: dict[str, Any]) -> str:
    dataset_name = str(config.get("name", "")).strip()
    explicit = config.get("split")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    return _resolve_spec(dataset_name).default_split


def resolve_normalized_dir(config: dict[str, Any]) -> Path:
    explicit = config.get("normalized_dir")
    if isinstance(explicit, str) and explicit.strip():
        return Path(explicit).expanduser().resolve()
    dataset_name = str(config.get("name", "default")).strip("/")
    return Path("data/normalized") / dataset_name


def try_load_normalized(config: dict[str, Any]) -> dict[str, Any] | None:
    dataset = get_dataset(config)
    if not dataset.exists():
        return None

    metadata = dataset.load_metadata()
    if not _is_compatible(metadata=metadata, config=config, split=dataset.split):
        return None

    payload = dataset.load_payload()
    payload.setdefault("metadata", {})
    payload["metadata"].update(
        {
            "dataset_name": str(config.get("name", metadata.get("dataset_name"))),
            "normalized_path": str(dataset.normalized_dir),
            "split": dataset.split,
            "loaded_from": "normalized",
            "normalized_schema_version": NORMALIZED_SCHEMA_VERSION,
        }
    )
    return payload


def process_dataset(
    config: dict[str, Any],
    *,
    force_rebuild: bool = False,
) -> dict[str, Any]:
    if not force_rebuild:
        reusable = try_load_normalized(config)
        if reusable is not None:
            return reusable

    dataset_name = str(config.get("name", "")).strip()
    spec = _resolve_spec(dataset_name)
    normalized_dir = resolve_normalized_dir(config)
    split = get_split(config)
    if not bool(config.get("download_if_missing", True)) and not normalized_dir.exists():
        raise FileNotFoundError(
            f"Normalized dataset not found at {normalized_dir} and download_if_missing=false."
        )

    if force_rebuild and normalized_dir.exists():
        shutil.rmtree(normalized_dir)

    if spec.kind == "script":
        _build_via_script(spec=spec, config=config, normalized_dir=normalized_dir, split=split)
    elif spec.kind == "pyterrier":
        _build_via_pyterrier(spec=spec, config=config, normalized_dir=normalized_dir, split=split)
    else:  # pragma: no cover
        raise ValueError(f"Unsupported dataset spec kind: {spec.kind}")

    payload = try_load_normalized(config)
    if payload is None:
        raise RuntimeError(
            f"Dataset '{dataset_name}' was prepared but the normalized cache is unreadable at {normalized_dir}"
        )
    payload["metadata"]["loaded_from"] = "built"
    return payload


def ensure_dataset_prepared(config: dict[str, Any], *, force_rebuild: bool = False) -> Path:
    payload = process_dataset(config, force_rebuild=force_rebuild)
    metadata = payload.get("metadata", {})
    normalized_path = metadata.get("normalized_path")
    if not isinstance(normalized_path, str) or not normalized_path:
        raise ValueError("Prepared dataset metadata is missing normalized_path.")
    return Path(normalized_path)


def _resolve_spec(dataset_name: str) -> DatasetSpec:
    if dataset_name in _SPECS:
        return _SPECS[dataset_name]
    if dataset_name.startswith("beir/"):
        dataset_id = dataset_name.split("/", 1)[1]
        if not dataset_id:
            raise ValueError("BEIR dataset name missing. Expected format 'beir/<dataset_name>'.")
        return DatasetSpec(
            kind="pyterrier",
            default_split="test",
            pyterrier_name=f"irds:beir/{dataset_id}",
        )
    raise ValueError(f"Unsupported dataset name for PyTerrier migration: {dataset_name}")


def _is_compatible(*, metadata: dict[str, Any], config: dict[str, Any], split: str) -> bool:
    if not isinstance(metadata, dict):
        return False
    if metadata.get("normalized_schema_version") != NORMALIZED_SCHEMA_VERSION:
        return False
    if metadata.get("dataset_name") != str(config.get("name", "")).strip():
        return False
    if metadata.get("split") != split:
        return False
    normalized_files = metadata.get("normalized_files")
    if not isinstance(normalized_files, dict):
        return False
    if normalized_files.get("corpus") != "corpus.jsonl":
        return False
    if normalized_files.get("topics") != "topics.jsonl":
        return False
    if normalized_files.get("qrels") != f"qrels/{split}.tsv":
        return False
    return True


def _build_via_script(
    *,
    spec: DatasetSpec,
    config: dict[str, Any],
    normalized_dir: Path,
    split: str,
) -> None:
    project_root = Path(__file__).resolve().parents[2]
    script_path = project_root / "scripts" / str(spec.script_name)
    if not script_path.exists():
        raise FileNotFoundError(f"Dataset builder script not found: {script_path}")

    normalized_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(script_path),
        "--source-split",
        str(config.get("source_split", split)),
        "--output-split",
        split,
        "--target-dir",
        str(normalized_dir),
    ]

    hf_dataset = config.get("hf_dataset")
    if isinstance(hf_dataset, str) and hf_dataset.strip():
        cmd.extend(["--hf-dataset", hf_dataset.strip()])

    docs_config = config.get("docs_config")
    if isinstance(docs_config, str) and docs_config.strip():
        cmd.extend(["--docs-config", docs_config.strip()])

    queries_config = config.get("queries_config")
    if isinstance(queries_config, str) and queries_config.strip():
        cmd.extend(["--queries-config", queries_config.strip()])

    qrels_config = config.get("qrels_config")
    if isinstance(qrels_config, str) and qrels_config.strip():
        cmd.extend(["--qrels-config", qrels_config.strip()])

    max_documents = config.get("max_documents")
    if isinstance(max_documents, int) and max_documents > 0:
        cmd.extend(["--max-documents", str(max_documents)])

    max_queries = config.get("max_queries")
    if isinstance(max_queries, int) and max_queries > 0:
        cmd.extend(["--max-queries", str(max_queries)])

    max_chars = config.get("max_chars_per_doc")
    if isinstance(max_chars, int) and max_chars > 0:
        cmd.extend(["--max-chars-per-doc", str(max_chars)])

    subprocess.run(cmd, cwd=project_root, check=True)


def _build_via_pyterrier(
    *,
    spec: DatasetSpec,
    config: dict[str, Any],
    normalized_dir: Path,
    split: str,
) -> None:
    if pt is None:
        raise ImportError(
            "pyterrier is not installed in this environment. Install it to prepare PyTerrier-managed datasets."
        )

    dataset_name = str(config.get("name", "")).strip()
    pt_name = str(spec.pyterrier_name or dataset_name)
    dataset = pt.get_dataset(pt_name)

    topics_df = _load_pt_topics(dataset=dataset, dataset_name=dataset_name, split=split)
    qrels_df = _load_pt_qrels(dataset=dataset, dataset_name=dataset_name, split=split)

    normalized_dir.mkdir(parents=True, exist_ok=True)
    (normalized_dir / "qrels").mkdir(parents=True, exist_ok=True)

    corpus_count = _stream_pt_corpus(
        dataset=dataset,
        dataset_name=dataset_name,
        output_path=normalized_dir / "corpus.jsonl",
    )
    _write_jsonl(normalized_dir / "topics.jsonl", topics_df.to_dict(orient="records"))
    qrels_df.to_csv(normalized_dir / "qrels" / f"{split}.tsv", sep="\t", index=False)

    metadata = {
        "dataset_name": dataset_name,
        "split": split,
        "normalized_schema_version": NORMALIZED_SCHEMA_VERSION,
        "normalized_path": str(normalized_dir),
        "source": "pyterrier",
        "pyterrier_dataset_name": pt_name,
        "num_documents": corpus_count,
        "num_queries": int(len(topics_df)),
        "num_qrels": int(len(qrels_df)),
        "normalized_files": {
            "corpus": "corpus.jsonl",
            "topics": "topics.jsonl",
            "qrels": f"qrels/{split}.tsv",
        },
    }
    (normalized_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _load_pt_topics(*, dataset: Any, dataset_name: str, split: str) -> pd.DataFrame:
    if dataset_name in {"msmarco", "msmarco-docs"}:
        topics = dataset.get_topics(split)
    else:
        topics = dataset.get_topics()
    frame = pd.DataFrame(topics).copy()
    if "qid" not in frame.columns:
        raise ValueError(f"PyTerrier topics for '{dataset_name}' do not contain a 'qid' column.")
    if "query" not in frame.columns:
        text_col = "text" if "text" in frame.columns else None
        if text_col is None:
            raise ValueError(f"PyTerrier topics for '{dataset_name}' do not contain a 'query' column.")
        frame = frame.rename(columns={text_col: "query"})
    return frame[["qid", "query"]].astype({"qid": str, "query": str})


def _load_pt_qrels(*, dataset: Any, dataset_name: str, split: str) -> pd.DataFrame:
    if dataset_name in {"msmarco", "msmarco-docs"}:
        qrels = dataset.get_qrels(split)
    else:
        qrels = dataset.get_qrels()
    frame = pd.DataFrame(qrels).copy()
    rename_map = {}
    if "docno" not in frame.columns and "doc_id" in frame.columns:
        rename_map["doc_id"] = "docno"
    if "label" not in frame.columns and "relevance" in frame.columns:
        rename_map["relevance"] = "label"
    if rename_map:
        frame = frame.rename(columns=rename_map)
    required = {"qid", "docno", "label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"PyTerrier qrels for '{dataset_name}' are missing columns: {sorted(missing)}")
    return frame[["qid", "docno", "label"]].astype({"qid": str, "docno": str, "label": int})


def _normalize_corpus_row(row: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise TypeError(f"Invalid corpus row from PyTerrier: expected dict, got {type(row).__name__}.")
    docno = (
        row.get("docno")
        or row.get("doc_id")
        or row.get("id")
        or row.get("_id")
    )
    if docno is None:
        raise ValueError("PyTerrier corpus row does not contain a document identifier.")
    normalized = dict(row)
    normalized["docno"] = str(docno)
    normalized.pop("doc_id", None)
    normalized.pop("id", None)
    normalized.pop("_id", None)
    return normalized


def _stream_pt_corpus(
    *,
    dataset: Any,
    dataset_name: str,
    output_path: Path,
) -> int:
    count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for row in dataset.get_corpus_iter(verbose=True):
            normalized = _normalize_corpus_row(row)
            f.write(json.dumps(normalized, ensure_ascii=False) + "\n")
            count += 1
    if count == 0:
        raise RuntimeError(f"PyTerrier returned an empty corpus for dataset '{dataset_name}'.")
    return count


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
