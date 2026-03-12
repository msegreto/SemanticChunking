from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseDatasetProcessor(ABC):
    """
    Base class for dataset processors.

    Flow:
    1. If normalized data exists and can be used -> load from normalized.
    2. Otherwise ensure raw data is available.
    3. Load from raw.
    4. Save normalized cache (if enabled).
    5. Return normalized in-memory representation.

    Normalized on-disk schema:
        data/normalized/<dataset_name>/
            documents.jsonl
            queries.json
            qrels/<split>.json
            metadata.json
    """

    DEFAULT_SPLIT = "default"
    NORMALIZED_SCHEMA_VERSION = "1.0"

    def process(self, config: dict) -> Any:
        split = self.get_split(config)
        normalized_dir = self.resolve_normalized_dir(config)

        if self.should_use_normalized(config) and self.normalized_exists(normalized_dir, split):
            data = self.load_normalized(normalized_dir=normalized_dir, split=split, config=config)
            data.setdefault("metadata", {})
            data["metadata"].update(
                {
                    "loaded_from": "normalized",
                    "normalized_path": str(normalized_dir),
                    "split": split,
                }
            )
            return data

        raw_path = self.ensure_available(config)
        data = self.load_raw(raw_path=raw_path, config=config)

        data.setdefault("metadata", {})
        data["metadata"].update(
            {
                "loaded_from": "raw",
                "raw_path": str(raw_path),
                "normalized_path": str(normalized_dir),
                "split": split,
                "normalized_schema_version": self.NORMALIZED_SCHEMA_VERSION,
            }
        )

        if self.should_save_normalized(config):
            self.save_normalized(data=data, normalized_dir=normalized_dir, split=split)
            data["metadata"].update(
                {
                    "normalized_files": {
                        "documents": "documents.jsonl",
                        "queries": "queries.json",
                        "qrels": f"qrels/{split}.json",
                    }
                }
            )

        return data

    def get_split(self, config: dict) -> str:
        return str(config.get("split", self.DEFAULT_SPLIT))

    def resolve_raw_dir(self, config: dict) -> Path:
        if config.get("data_dir"):
            return Path(config["data_dir"]).expanduser().resolve()

        dataset_name = str(config.get("name", "default")).strip("/")
        return Path("data/raw") / dataset_name

    def resolve_normalized_dir(self, config: dict) -> Path:
        if config.get("normalized_dir"):
            return Path(config["normalized_dir"]).expanduser().resolve()

        dataset_name = str(config.get("name", "default")).strip("/")
        return Path("data/normalized") / dataset_name

    def should_download(self, config: dict) -> bool:
        return bool(config.get("download_if_missing", True))

    def should_use_normalized(self, config: dict) -> bool:
        # If max_documents is set, avoid accidentally reusing a full-cache normalized
        # for a partial/debug run.
        if config.get("max_documents") is not None:
            return False
        if bool(config.get("force_rebuild_normalized", False)):
            return False
        return bool(config.get("use_normalized_if_available", True))

    def should_save_normalized(self, config: dict) -> bool:
        if config.get("max_documents") is not None:
            return False
        return bool(config.get("save_normalized", True))

    def normalized_exists(self, normalized_dir: Path, split: str) -> bool:
        documents_path = normalized_dir / "documents.jsonl"
        queries_path = normalized_dir / "queries.json"
        qrels_path = normalized_dir / "qrels" / f"{split}.json"
        metadata_path = normalized_dir / "metadata.json"

        return (
            documents_path.exists()
            and queries_path.exists()
            and qrels_path.exists()
            and metadata_path.exists()
        )

    def load_normalized(self, normalized_dir: Path, split: str, config: dict) -> Any:
        documents_path = normalized_dir / "documents.jsonl"
        queries_path = normalized_dir / "queries.json"
        qrels_path = normalized_dir / "qrels" / f"{split}.json"
        metadata_path = normalized_dir / "metadata.json"

        if not self.normalized_exists(normalized_dir, split):
            raise FileNotFoundError(
                f"Normalized dataset not complete for split='{split}' at '{normalized_dir}'"
            )

        documents = self._read_documents_jsonl(documents_path)
        queries = json.loads(queries_path.read_text(encoding="utf-8"))
        qrels = json.loads(qrels_path.read_text(encoding="utf-8"))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        metadata.update(
            {
                "dataset_name": config.get("name", metadata.get("dataset_name")),
                "split": split,
            }
        )

        return {
            "documents": documents,
            "queries": queries,
            "qrels": qrels,
            "metadata": metadata,
        }

    def save_normalized(self, data: dict, normalized_dir: Path, split: str) -> None:
        documents = data.get("documents", {})
        queries = data.get("queries", {})
        qrels = data.get("qrels", {})
        metadata = dict(data.get("metadata", {}))

        if documents is None:
            raise ValueError(
                "Cannot save normalized dataset because 'documents' is None. "
                "For now, processors should fully load documents before normalization."
            )

        self._ensure_dir(normalized_dir)
        self._ensure_dir(normalized_dir / "qrels")

        documents_path = normalized_dir / "documents.jsonl"
        queries_path = normalized_dir / "queries.json"
        qrels_path = normalized_dir / "qrels" / f"{split}.json"
        metadata_path = normalized_dir / "metadata.json"

        self._write_documents_jsonl(documents_path, documents)
        queries_path.write_text(json.dumps(queries, ensure_ascii=False, indent=2), encoding="utf-8")
        qrels_path.write_text(json.dumps(qrels, ensure_ascii=False, indent=2), encoding="utf-8")

        metadata.update(
            {
                "normalized_schema_version": self.NORMALIZED_SCHEMA_VERSION,
                "normalized_files": {
                    "documents": "documents.jsonl",
                    "queries": "queries.json",
                    "qrels": f"qrels/{split}.json",
                },
            }
        )
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _ensure_dir(path: Path) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _write_documents_jsonl(path: Path, documents: dict[str, dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as file_obj:
            for doc_id, payload in documents.items():
                row = {"id": doc_id, **payload}
                file_obj.write(json.dumps(row, ensure_ascii=False) + "\n")

    @staticmethod
    def _read_documents_jsonl(path: Path) -> dict[str, dict[str, Any]]:
        documents: dict[str, dict[str, Any]] = {}
        with path.open("r", encoding="utf-8") as file_obj:
            for line in file_obj:
                if not line.strip():
                    continue
                row = json.loads(line)
                doc_id = row.pop("id")
                documents[doc_id] = row
        return documents

    @abstractmethod
    def ensure_available(self, config: dict) -> Path:
        raise NotImplementedError

    @abstractmethod
    def load_raw(self, raw_path: Path, config: dict) -> Any:
        raise NotImplementedError