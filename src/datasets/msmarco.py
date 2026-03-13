from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from src.datasets.base import BaseDatasetProcessor
from src.datasets.download_utils import (
    download_msmarco_documents,
    list_missing_msmarco_files,
    validate_msmarco_raw,
)


class MSMarcoDatasetProcessor(BaseDatasetProcessor):
    DEFAULT_SPLIT = "test"
    _SUPPORTED_SPLITS = {"train", "dev", "test"}
    _ALIASES = {"msmarco", "msmarco-docs"}

    def _canonical_dataset_name(self, config: dict) -> str:
        requested = str(config.get("name", "msmarco")).strip().lower()
        if requested not in self._ALIASES:
            return requested
        return "msmarco-docs"

    def expected_dataset_names(self, config: dict) -> set[str]:
        requested = str(config.get("name", "msmarco")).strip()
        return {requested, self._canonical_dataset_name(config)}

    def ensure_available(self, config: dict) -> Path:
        split = self.get_split(config)
        if split not in self._SUPPORTED_SPLITS:
            raise ValueError(f"Unsupported MS MARCO split '{split}'")

        raw_path = self.resolve_raw_dir(config)
        raw_path.mkdir(parents=True, exist_ok=True)

        if not validate_msmarco_raw(raw_path, split):
            if not self.should_download(config):
                missing = list_missing_msmarco_files(raw_path, split)
                raise FileNotFoundError(
                    f"MS MARCO dataset incomplete at '{raw_path}'. Missing files for split '{split}': {missing}"
                )

            download_msmarco_documents(
                target_dir=raw_path,
                force=bool(config.get("force_download", False)),
            )

            if not validate_msmarco_raw(raw_path, split):
                missing = list_missing_msmarco_files(raw_path, split)
                raise FileNotFoundError(
                    f"MS MARCO download finished, but raw dataset is still incomplete. Missing: {missing}"
                )

        return raw_path

    def load_raw(self, raw_path: Path, config: dict) -> Any:
        split = self.get_split(config)
        requested_name = str(config.get("name", "msmarco")).strip()
        canonical_name = self._canonical_dataset_name(config)
        max_documents = config.get("max_documents")

        corpus_path = raw_path / "msmarco-docs.tsv"
        queries_path = self._resolve_queries_path(raw_path, split)
        qrels_path = self._resolve_qrels_path(raw_path, split)

        documents = self._load_corpus(corpus_path, max_documents=max_documents)
        queries = self._load_queries(queries_path)
        qrels = self._load_qrels(qrels_path) if qrels_path.exists() else {}

        return {
            "documents": documents,
            "queries": queries,
            "qrels": qrels,
            "metadata": {
                "dataset_name": requested_name,
                "canonical_dataset_name": canonical_name,
                "processor": "msmarco-docs",
                "split": split,
                "source": "official_msmarco",
                "raw_path": str(raw_path),
                "documents_path": str(corpus_path),
                "queries_path": str(queries_path),
                "qrels_path": str(qrels_path) if qrels_path.exists() else None,
                "max_documents": max_documents,
                "num_documents": len(documents),
                "num_queries": len(queries),
                "num_qrels_queries": len(qrels),
            },
        }

    @staticmethod
    def _resolve_queries_path(raw_path: Path, split: str) -> Path:
        if split == "train":
            return raw_path / "msmarco-doctrain-queries.tsv"
        if split == "dev":
            return raw_path / "msmarco-docdev-queries.tsv"
        if split == "test":
            return raw_path / "docleaderboard-queries.tsv"
        raise ValueError(f"Unsupported MS MARCO split '{split}'")

    @staticmethod
    def _resolve_qrels_path(raw_path: Path, split: str) -> Path:
        if split == "train":
            return raw_path / "msmarco-doctrain-qrels.tsv"
        if split == "dev":
            return raw_path / "msmarco-docdev-qrels.tsv"
        if split == "test":
            return raw_path / "msmarco-doctest-qrels.tsv"
        raise ValueError(f"Unsupported MS MARCO split '{split}'")

    @staticmethod
    def _load_corpus(
        corpus_path: Path,
        max_documents: int | None = None,
    ) -> dict[str, dict[str, str]]:
        documents: dict[str, dict[str, str]] = {}

        with corpus_path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.rstrip("\n")
                if not line:
                    continue

                parts = line.split("\t", 3)
                if len(parts) != 4:
                    raise ValueError(
                        f"Invalid MS MARCO document row in '{corpus_path}': expected 4 tab-separated fields."
                    )

                doc_id, url, title, text = parts
                documents[doc_id] = {
                    "url": url,
                    "title": title,
                    "text": text,
                }

                if max_documents is not None and idx + 1 >= max_documents:
                    break

        return documents

    @staticmethod
    def _load_queries(queries_path: Path) -> dict[str, str]:
        queries: dict[str, str] = {}

        with queries_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue

                parts = line.split("\t", 1)
                if len(parts) != 2:
                    raise ValueError(
                        f"Invalid MS MARCO query row in '{queries_path}': expected 2 tab-separated fields."
                    )

                query_id, text = parts
                queries[str(query_id)] = text

        return queries

    @staticmethod
    def _load_qrels(qrels_path: Path) -> dict[str, dict[str, int]]:
        qrels: dict[str, dict[str, int]] = {}

        with qrels_path.open("r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if not row:
                    continue

                if len(row) >= 4:
                    query_id = str(row[0])
                    doc_id = str(row[2])
                    score = int(row[3])
                elif len(row) == 3:
                    query_id = str(row[0])
                    doc_id = str(row[1])
                    score = int(row[2])
                else:
                    raise ValueError(
                        f"Invalid MS MARCO qrels row in '{qrels_path}': expected at least 3 tab-separated fields."
                    )

                qrels.setdefault(query_id, {})[doc_id] = score

        return qrels
