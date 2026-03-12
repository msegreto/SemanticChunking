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


class MSMarcoDatasetProcessor(BaseDatasetProcessor):
    DEFAULT_SPLIT = "test"
    _SUPPORTED_SPLITS = {"train", "dev", "test"}

    def ensure_available(self, config: dict) -> Path:
        split = self.get_split(config)
        if split not in self._SUPPORTED_SPLITS:
            raise ValueError(f"Unsupported MS MARCO split '{split}'")

        raw_path = self.resolve_raw_dir(config)
        raw_path.mkdir(parents=True, exist_ok=True)

        if not validate_beir_raw(raw_path, split):
            # Scarica la versione BEIR "msmarco" dentro raw_path.parent/msmarco
            downloaded_dir = download_beir_dataset(
                dataset_name="msmarco",
                target_root=raw_path.parent,
                force=False,
            )

            if not validate_beir_raw(downloaded_dir, split):
                missing = list_missing_beir_files(downloaded_dir, split)
                raise FileNotFoundError(
                    f"BEIR MSMARCO download finished, but raw dataset is still incomplete. Missing: {missing}"
                )
            return downloaded_dir

        return raw_path

    def load_raw(self, raw_path: Path, config: dict) -> Any:
        split = self.get_split(config)

        corpus_path = raw_path / "corpus.jsonl"
        queries_path = raw_path / "queries.jsonl"
        qrels_path = raw_path / "qrels" / f"{split}.tsv"

        documents = self._load_corpus(corpus_path)
        queries = self._load_queries(queries_path)
        qrels = self._load_qrels(qrels_path) if qrels_path.exists() else {}

        return {
            "documents": documents,
            "queries": queries,
            "qrels": qrels,
            "metadata": {
                "dataset_name": "beir/msmarco",
                "processor": "msmarco",
                "split": split,
                "source": "beir",
                "raw_path": str(raw_path),
                "num_documents": len(documents),
                "num_queries": len(queries),
                "num_qrels_queries": len(qrels),
            },
        }

    @staticmethod
    def _load_corpus(corpus_path: Path) -> dict[str, dict[str, str]]:
        documents: dict[str, dict[str, str]] = {}

        with corpus_path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                doc_id = str(row["_id"])
                documents[doc_id] = {
                    "url": "",
                    "title": row.get("title", "") or "",
                    "text": row.get("text", "") or "",
                }

        return documents

    @staticmethod
    def _load_queries(queries_path: Path) -> dict[str, str]:
        queries: dict[str, str] = {}

        with queries_path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                query_id = str(row["_id"])
                queries[query_id] = row.get("text", "") or ""

        return queries

    @staticmethod
    def _load_qrels(qrels_path: Path) -> dict[str, dict[str, int]]:
        qrels: dict[str, dict[str, int]] = {}

        with qrels_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                query_id = str(row["query-id"])
                doc_id = str(row["corpus-id"])
                score = int(row["score"])
                qrels.setdefault(query_id, {})[doc_id] = score

        return qrels