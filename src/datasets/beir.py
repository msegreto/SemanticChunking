from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from src.datasets.base import BaseDatasetProcessor
from src.datasets.download_utils import (
    download_beir_dataset,
    list_missing_beir_files,
    validate_beir_raw,
)


class BEIRDatasetProcessor(BaseDatasetProcessor):
    DEFAULT_SPLIT = "test"

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name

    def ensure_available(self, config: dict) -> Path:
        split = self.get_split(config)
        raw_path = self.resolve_raw_dir(config)

        if validate_beir_raw(raw_path, split):
            return raw_path

        if not self.should_download(config):
            missing = list_missing_beir_files(raw_path, split)
            raise FileNotFoundError(
                f"BEIR dataset incomplete at '{raw_path}'. Missing files for split '{split}': {missing}"
            )

        download_beir_dataset(dataset_name=self.dataset_name, target_root=raw_path.parent)

        if not validate_beir_raw(raw_path, split):
            missing = list_missing_beir_files(raw_path, split)
            raise FileNotFoundError(
                f"BEIR download finished, but raw dataset is still incomplete. Missing: {missing}"
            )

        return raw_path

    def load_raw(self, raw_path: Path, config: dict) -> Any:
        split = self.get_split(config)
        max_documents = config.get("max_documents")

        corpus_path = raw_path / "corpus.jsonl"
        queries_path = raw_path / "queries.jsonl"
        qrels_path = raw_path / "qrels" / f"{split}.tsv"

        documents = self._load_corpus(corpus_path, max_documents=max_documents)
        queries = self._load_queries(queries_path)
        qrels = self._load_qrels(qrels_path)

        return {
            "documents": documents,
            "queries": queries,
            "qrels": qrels,
            "metadata": {
                "dataset_name": f"beir/{self.dataset_name}",
                "processor": "beir",
                "split": split,
                "documents_path": str(corpus_path),
                "queries_path": str(queries_path),
                "qrels_path": str(qrels_path),
                "max_documents": max_documents,
                "num_documents": len(documents),
                "num_queries": len(queries),
                "num_qrels_queries": len(qrels),
            },
        }

    @staticmethod
    def _load_corpus(corpus_path: Path, max_documents: int | None = None) -> dict[str, dict[str, str]]:
        documents: dict[str, dict[str, str]] = {}

        with corpus_path.open("r", encoding="utf-8") as file_obj:
            for idx, line in enumerate(file_obj):
                if not line.strip():
                    continue

                record = json.loads(line)
                doc_id = record["_id"]
                documents[doc_id] = {
                    "title": record.get("title", ""),
                    "text": record.get("text", ""),
                }

                if max_documents is not None and idx + 1 >= max_documents:
                    break

        return documents

    @staticmethod
    def _load_queries(queries_path: Path) -> dict[str, str]:
        queries: dict[str, str] = {}

        with queries_path.open("r", encoding="utf-8") as file_obj:
            for line in file_obj:
                if not line.strip():
                    continue

                record = json.loads(line)
                queries[record["_id"]] = record["text"]

        return queries

    @staticmethod
    def _load_qrels(qrels_path: Path) -> dict[str, dict[str, int]]:
        qrels: dict[str, dict[str, int]] = {}

        with qrels_path.open("r", encoding="utf-8") as file_obj:
            reader = csv.DictReader(file_obj, delimiter="\t")
            for row in reader:
                query_id = row["query-id"]
                doc_id = row["corpus-id"]
                score = int(row["score"])
                qrels.setdefault(query_id, {})[doc_id] = score

        return qrels