from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import spacy
from tqdm.auto import tqdm

from src.splitting.base import BaseSplitter


class SentenceSplitter(BaseSplitter):
    MAX_CHARS_PER_BATCH = 300_000

    def __init__(self):
        self._nlp_cache = {}

    def split(self, dataset_output: Dict[str, Any], config: dict) -> Dict[str, Any]:
        print(f"[SPLIT] sentence split with config={config}")

        model_name = config.get("model", "en_core_web_sm")
        metadata = dataset_output["metadata"]
        documents_path = Path(metadata["normalized_path"]) / metadata["normalized_files"]["documents"]
        output_file = self.resolve_output_path(config)

        nlp = self._load_spacy_model(model_name)
        max_chars_per_batch = min(
            self.MAX_CHARS_PER_BATCH,
            max(1, int(getattr(nlp, "max_length", 1_000_000)) - 1),
        )

        split_units: List[Dict[str, Any]] = []
        doc_stats: List[Dict[str, Any]] = []

        unit_id = 0

        with open(documents_path, "r", encoding="utf-8") as f:
            for line in f:
                doc = json.loads(line)

                doc_id = doc["id"]
                text = doc["text"]

                if not text or not str(text).strip():
                    doc_stats.append(
                        {
                            "doc_id": doc_id,
                            "num_sentences": 0,
                        }
                    )
                    continue

                sentence_count = 0
                for batch_start, batch_text in self._split_text_in_batches(text, max_chars_per_batch):
                    parsed = nlp(batch_text)

                    for sent in parsed.sents:
                        sent_text = sent.text.strip()

                        if not sent_text:
                            continue

                        split_units.append(
                            {
                                "unit_id": unit_id,
                                "doc_id": doc_id,
                                "sentence_idx": sentence_count,
                                "text": sent_text,
                                "char_start": batch_start + sent.start_char,
                                "char_end": batch_start + sent.end_char,
                            }
                        )

                        unit_id += 1
                        sentence_count += 1

                doc_stats.append(
                    {
                        "doc_id": doc_id,
                        "num_sentences": sentence_count,
                    }
                )

        return self._build_split_output(
            split_units=split_units,
            doc_stats=doc_stats,
            metadata=metadata,
            model_name=model_name,
            output_file=output_file,
            reused_existing_split=False,
        )

    def split_stream(self, dataset_output: Dict[str, Any], config: dict) -> Dict[str, Any]:
        """
        Streaming variant:
        - does not accumulate split units in memory
        - writes units directly to output JSONL
        - returns only aggregate metadata + per-document sentence counts
        """
        print(f"[SPLIT] sentence split (streaming) with config={config}")

        model_name = config.get("model", "en_core_web_sm")
        metadata = dataset_output["metadata"]
        documents_path = Path(metadata["normalized_path"]) / metadata["normalized_files"]["documents"]
        output_file = self.resolve_output_path(config)
        if output_file is None:
            raise ValueError("Streaming sentence split requires 'split.output_path' to be set.")

        save_output = bool(config.get("save_output", False))
        if not save_output:
            raise ValueError("Streaming sentence split requires 'split.save_output: true'.")

        if output_file.exists() and output_file.stat().st_size > 0:
            print(f"[SPLIT] reusing existing sentence split from {output_file}")
            num_units = self._count_non_empty_lines(output_file)
            num_documents = metadata.get("num_documents")
            if not isinstance(num_documents, int) or num_documents < 0:
                num_documents = 0
            return {
                "split_units_path": str(output_file),
                "documents": [],
                "metadata": {
                    "split_type": "sentence",
                    "spacy_model": model_name,
                    "num_documents": int(num_documents),
                    "num_units": num_units,
                    "source_metadata": metadata,
                    "reused_existing_split": True,
                    "split_path": str(output_file),
                    "streaming": True,
                },
            }

        output_file.parent.mkdir(parents=True, exist_ok=True)

        nlp = self._load_spacy_model(model_name)
        max_chars_per_batch = min(
            self.MAX_CHARS_PER_BATCH,
            max(1, int(getattr(nlp, "max_length", 1_000_000)) - 1),
        )

        unit_id = 0
        doc_count = 0

        expected_docs = metadata.get("num_documents")
        progress = tqdm(
            total=int(expected_docs) if isinstance(expected_docs, int) and expected_docs > 0 else None,
            desc="Split (streaming)",
            unit="doc",
            dynamic_ncols=True,
            miniters=1,
        )
        with open(documents_path, "r", encoding="utf-8") as src, open(output_file, "w", encoding="utf-8") as dst:
            for line in src:
                doc = json.loads(line)

                doc_id = doc["id"]
                text = doc["text"]
                doc_count += 1

                if not text or not str(text).strip():
                    progress.update(1)
                    progress.set_postfix(docs=doc_count, units=unit_id, refresh=False)
                    continue

                sentence_count = 0
                for batch_start, batch_text in self._split_text_in_batches(text, max_chars_per_batch):
                    parsed = nlp(batch_text)
                    for sent in parsed.sents:
                        sent_text = sent.text.strip()
                        if not sent_text:
                            continue

                        row = {
                            "unit_id": unit_id,
                            "doc_id": doc_id,
                            "sentence_idx": sentence_count,
                            "text": sent_text,
                            "char_start": batch_start + sent.start_char,
                            "char_end": batch_start + sent.end_char,
                        }
                        dst.write(json.dumps(row, ensure_ascii=False) + "\n")
                        unit_id += 1
                        sentence_count += 1

                progress.update(1)
                progress.set_postfix(docs=doc_count, units=unit_id, refresh=False)
        progress.close()

        return {
            "split_units_path": str(output_file),
            "documents": [],
            "metadata": {
                "split_type": "sentence",
                "spacy_model": model_name,
                "num_documents": doc_count,
                "num_units": unit_id,
                "source_metadata": metadata,
                "reused_existing_split": False,
                "split_path": str(output_file),
                "streaming": True,
            },
        }

    def build_streaming_components(self, config: dict) -> dict[str, Any]:
        model_name = config.get("model", "en_core_web_sm")
        nlp = self._load_spacy_model(model_name)
        max_chars_per_batch = min(
            self.MAX_CHARS_PER_BATCH,
            max(1, int(getattr(nlp, "max_length", 1_000_000)) - 1),
        )
        return {
            "model_name": model_name,
            "nlp": nlp,
            "max_chars_per_batch": max_chars_per_batch,
        }

    def split_document_streaming(
        self,
        *,
        doc_id: str,
        text: str,
        unit_id_start: int,
        nlp: Any,
        max_chars_per_batch: int,
    ) -> tuple[list[dict[str, Any]], int]:
        if not text or not str(text).strip():
            return [], unit_id_start

        units: list[dict[str, Any]] = []
        next_unit_id = unit_id_start
        sentence_idx = 0
        for batch_start, batch_text in self._split_text_in_batches(text, max_chars_per_batch):
            parsed = nlp(batch_text)
            for sent in parsed.sents:
                sent_text = sent.text.strip()
                if not sent_text:
                    continue
                units.append(
                    {
                        "unit_id": next_unit_id,
                        "doc_id": doc_id,
                        "sentence_idx": sentence_idx,
                        "text": sent_text,
                        "char_start": batch_start + sent.start_char,
                        "char_end": batch_start + sent.end_char,
                    }
                )
                next_unit_id += 1
                sentence_idx += 1

        return units, next_unit_id

    def try_load_reusable_output(self, dataset_output: Dict[str, Any], config: dict) -> Dict[str, Any] | None:
        save_output = config.get("save_output", False)
        output_file = self.resolve_output_path(config)
        if not save_output or output_file is None:
            return None
        if not output_file.exists() or output_file.stat().st_size <= 0:
            return None

        print(f"[SPLIT] reusing existing sentence split from {output_file}")
        split_units = self._load_existing_split(output_file)
        return self._build_split_output(
            split_units=split_units,
            doc_stats=None,
            metadata=dataset_output["metadata"],
            model_name=config.get("model", "en_core_web_sm"),
            output_file=output_file,
            reused_existing_split=True,
        )

    def save_output(self, split_output: Dict[str, Any], config: dict) -> None:
        save_output = config.get("save_output", False)
        output_file = self.resolve_output_path(config)
        if not save_output or output_file is None:
            return

        output_file.parent.mkdir(parents=True, exist_ok=True)
        split_units = split_output.get("split_units", [])
        with open(output_file, "w", encoding="utf-8") as f:
            for item in split_units:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def _build_split_output(
        self,
        *,
        split_units: List[Dict[str, Any]],
        doc_stats: List[Dict[str, Any]] | None,
        metadata: Dict[str, Any],
        model_name: str,
        output_file: Path | None,
        reused_existing_split: bool,
    ) -> Dict[str, Any]:
        if doc_stats is None:
            doc_stats = self._build_doc_stats(split_units)
        return {
            "split_units": split_units,
            "documents": doc_stats,
            "metadata": {
                "split_type": "sentence",
                "spacy_model": model_name,
                "num_documents": len(doc_stats),
                "num_units": len(split_units),
                "source_metadata": metadata,
                "reused_existing_split": reused_existing_split,
                "split_path": str(output_file) if output_file else None,
            },
        }

    def _load_existing_split(self, output_file: Path) -> List[Dict[str, Any]]:
        split_units: List[Dict[str, Any]] = []

        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                split_units.append(json.loads(line))

        return split_units

    def _build_doc_stats(self, split_units: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        counts: Dict[str, int] = {}

        for unit in split_units:
            doc_id = unit["doc_id"]
            counts[doc_id] = counts.get(doc_id, 0) + 1

        return [
            {
                "doc_id": doc_id,
                "num_sentences": num_sentences,
            }
            for doc_id, num_sentences in counts.items()
        ]

    def _load_spacy_model(self, model_name: str):

        if model_name in self._nlp_cache:
            return self._nlp_cache[model_name]

        try:
            nlp = spacy.load(model_name)

        except Exception:
            nlp = spacy.blank("en")
            nlp.add_pipe("sentencizer")

        self._nlp_cache[model_name] = nlp

        return nlp

    def _split_text_in_batches(self, text: str, max_chars: int) -> List[tuple[int, str]]:
        if len(text) <= max_chars:
            return [(0, text)]

        batches: List[tuple[int, str]] = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = min(start + max_chars, text_length)
            if end < text_length:
                boundary = text.rfind(" ", start, end)
                if boundary > start:
                    end = boundary

            chunk = text[start:end]
            if not chunk:
                end = min(start + max_chars, text_length)
                chunk = text[start:end]
                if not chunk:
                    break

            batches.append((start, chunk))
            start = end

            while start < text_length and text[start].isspace():
                start += 1

        return batches

    def iter_split_units(self, split_path: Path) -> Iterable[Dict[str, Any]]:
        with split_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)

    def _count_non_empty_lines(self, output_file: Path) -> int:
        count = 0
        with output_file.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
        return count
