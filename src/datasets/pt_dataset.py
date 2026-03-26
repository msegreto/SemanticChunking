from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

try:
    import pyterrier as pt
except ImportError:  # pragma: no cover
    pt = None


_PtDatasetBase = pt.datasets.Dataset if pt is not None else object


class PyTerrierNormalizedDataset(_PtDatasetBase):
    def __init__(self, *, dataset_name: str, normalized_dir: Path, split: str) -> None:
        self.dataset_name = str(dataset_name).strip()
        self.normalized_dir = Path(normalized_dir)
        self.split = str(split).strip()

    @property
    def corpus_path(self) -> Path:
        return self.normalized_dir / "corpus.jsonl"

    @property
    def topics_path(self) -> Path:
        return self.normalized_dir / "topics.jsonl"

    @property
    def qrels_path(self) -> Path:
        return self.normalized_dir / "qrels" / f"{self.split}.tsv"

    @property
    def metadata_path(self) -> Path:
        return self.normalized_dir / "metadata.json"

    def exists(self) -> bool:
        return (
            self.corpus_path.exists()
            and self.topics_path.exists()
            and self.qrels_path.exists()
            and self.metadata_path.exists()
        )

    def load_metadata(self) -> dict[str, Any]:
        return json.loads(self.metadata_path.read_text(encoding="utf-8"))

    def get_corpus_iter(self, *, verbose: bool = True) -> Iterable[dict[str, Any]]:
        del verbose
        with self.corpus_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)

    def get_topics(self, variant: str | None = None) -> pd.DataFrame:
        del variant
        return pd.DataFrame(self._read_jsonl(self.topics_path))

    def get_qrels(self, variant: str | None = None) -> pd.DataFrame:
        qrels_path = self.normalized_dir / "qrels" / f"{variant or self.split}.tsv"
        if not qrels_path.exists():
            raise FileNotFoundError(f"Qrels file not found: {qrels_path}")
        return pd.read_csv(qrels_path, sep="\t")

    def get_corpus_lang(self) -> str | None:
        metadata = self.load_metadata()
        value = metadata.get("corpus_lang")
        return str(value) if isinstance(value, str) and value.strip() else None

    def get_corpus(self) -> list[str]:
        return [str(self.corpus_path)]

    def load_payload(self) -> dict[str, Any]:
        metadata = self.load_metadata()
        payload: dict[str, Any] = {
            "corpus": self._read_jsonl(self.corpus_path),
            "topics": self._read_jsonl(self.topics_path),
            "qrels": self._read_qrels_tsv(self.qrels_path),
            "metadata": metadata,
            "pt_dataset": self,
        }

        evidences_path = self.normalized_dir / "evidences.json"
        answers_path = self.normalized_dir / "answers.json"
        if evidences_path.exists():
            payload["evidences"] = json.loads(evidences_path.read_text(encoding="utf-8"))
        if answers_path.exists():
            payload["answers"] = json.loads(answers_path.read_text(encoding="utf-8"))
        return payload

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    @staticmethod
    def _read_qrels_tsv(path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                qid = row.get("qid")
                docno = row.get("docno")
                label = row.get("label")
                if qid is None or docno is None or label is None:
                    continue
                rows.append({"qid": str(qid), "docno": str(docno), "label": int(label)})
        return rows
