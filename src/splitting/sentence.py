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
        save_output = config.get("save_output", False)
        output_path = config.get("output_path")

        metadata = dataset_output["metadata"]

        documents_path = (
            Path(metadata["normalized_path"])
            / metadata["normalized_files"]["documents"]
        )

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

        if save_output and output_path:

            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, "w", encoding="utf-8") as f:
                for item in split_units:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

        return {
            "split_units": split_units,
            "documents": doc_stats,
            "metadata": {
                "split_type": "sentence",
                "spacy_model": model_name,
                "num_documents": len(doc_stats),
                "num_units": len(split_units),
                "source_metadata": metadata,
            },
        }

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