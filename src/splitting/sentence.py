from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import spacy

from src.splitting.base import BaseSplitter


class SentenceSplitter(BaseSplitter):

    def __init__(self):
        self._nlp_cache = {}

    def split(self, dataset_output: Dict[str, Any], config: dict) -> Dict[str, Any]:
        print(f"[SPLIT] sentence split with config={config}")

        model_name = config.get("model", "en_core_web_sm")
        metadata = dataset_output["metadata"]
        documents_path = Path(metadata["normalized_path"]) / metadata["normalized_files"]["documents"]
        output_file = self.resolve_output_path(config)

        nlp = self._load_spacy_model(model_name)

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

                parsed = nlp(text)

                sentence_count = 0

                for sent_idx, sent in enumerate(parsed.sents):
                    sent_text = sent.text.strip()

                    if not sent_text:
                        continue

                    split_units.append(
                        {
                            "unit_id": unit_id,
                            "doc_id": doc_id,
                            "sentence_idx": sent_idx,
                            "text": sent_text,
                            "char_start": sent.start_char,
                            "char_end": sent.end_char,
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
