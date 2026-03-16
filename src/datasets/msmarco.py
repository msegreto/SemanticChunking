from __future__ import annotations

import gzip
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
        include_lookup = bool(config.get("include_lookup", False))
        extract_archives = bool(config.get("extract_archives", False))

        if not validate_msmarco_raw(raw_path, split, include_lookup=include_lookup):
            if not self.should_download(config):
                missing = list_missing_msmarco_files(raw_path, split, include_lookup=include_lookup)
                raise FileNotFoundError(
                    f"MS MARCO dataset incomplete at '{raw_path}'. Missing files for split '{split}': {missing}"
                )

            download_msmarco_documents(
                target_dir=raw_path,
                split=split,
                force=bool(config.get("force_download", False)),
                include_lookup=include_lookup,
                extract_archives=extract_archives,
            )

            if not validate_msmarco_raw(raw_path, split, include_lookup=include_lookup):
                missing = list_missing_msmarco_files(raw_path, split, include_lookup=include_lookup)
                raise FileNotFoundError(
                    f"MS MARCO download finished, but raw dataset is still incomplete. Missing: {missing}"
                )

        return raw_path

    def load_raw(self, raw_path: Path, config: dict) -> Any:
        split = self.get_split(config)
        requested_name = str(config.get("name", "msmarco")).strip()
        canonical_name = self._canonical_dataset_name(config)
        max_documents = config.get("max_documents")

        corpus_path = self._resolve_existing_path(raw_path, "msmarco-docs.tsv")
        queries_path = self._resolve_existing_path(raw_path, self._resolve_queries_filename(split))
        qrels_filename = self._resolve_qrels_filename(split)
        qrels_path = self._resolve_existing_path(raw_path, qrels_filename) if qrels_filename else None

        documents = self._load_corpus(corpus_path, max_documents=max_documents)
        queries = self._load_queries(queries_path)
        qrels = self._load_qrels(qrels_path) if qrels_path is not None else {}

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
                "qrels_path": str(qrels_path) if qrels_path is not None else None,
                "max_documents": max_documents,
                "num_documents": len(documents),
                "num_queries": len(queries),
                "num_qrels_queries": len(qrels),
            },
        }

    @staticmethod
    def _resolve_queries_filename(split: str) -> str:
        if split == "train":
            return "msmarco-doctrain-queries.tsv"
        if split == "dev":
            return "msmarco-docdev-queries.tsv"
        if split == "test":
            return "docleaderboard-queries.tsv"
        raise ValueError(f"Unsupported MS MARCO split '{split}'")

    @staticmethod
    def _resolve_qrels_filename(split: str) -> str | None:
        if split == "train":
            return "msmarco-doctrain-qrels.tsv"
        if split == "dev":
            return "msmarco-docdev-qrels.tsv"
        if split == "test":
            return None
        raise ValueError(f"Unsupported MS MARCO split '{split}'")

    @staticmethod
    def _resolve_existing_path(raw_path: Path, filename: str) -> Path:
        plain = raw_path / filename
        gz = raw_path / f"{filename}.gz"
        if plain.exists():
            return plain
        if gz.exists():
            return gz
        raise FileNotFoundError(
            f"Expected MS MARCO file not found at '{plain}' or '{gz}'."
        )

    @staticmethod
    def _open_text(path: Path):
        if path.suffix == ".gz":
            return gzip.open(path, "rt", encoding="utf-8")
        return path.open("r", encoding="utf-8")

    @staticmethod
    def _load_corpus(
        corpus_path: Path,
        max_documents: int | None = None,
    ) -> dict[str, dict[str, str]]:
        documents: dict[str, dict[str, str]] = {}

        with MSMarcoDatasetProcessor._open_text(corpus_path) as f:
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

        with MSMarcoDatasetProcessor._open_text(queries_path) as f:
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

        with MSMarcoDatasetProcessor._open_text(qrels_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # MS MARCO qrels can be distributed as whitespace-separated files
                # even when the extension is .tsv/.tsv.gz.
                fields = line.split()
                if len(fields) >= 4:
                    query_id = str(fields[0])
                    doc_id = str(fields[2])
                    score = int(fields[3])
                elif len(fields) == 3:
                    query_id = str(fields[0])
                    doc_id = str(fields[1])
                    score = int(fields[2])
                else:
                    raise ValueError(
                        f"Invalid MS MARCO qrels row in '{qrels_path}': expected at least 3 whitespace-separated fields."
                    )

                qrels.setdefault(query_id, {})[doc_id] = score

        return qrels
