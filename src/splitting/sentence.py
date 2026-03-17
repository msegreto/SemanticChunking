from __future__ import annotations

from typing import Any

import spacy

from src.splitting.base import BaseSplitter


class SentenceSplitter(BaseSplitter):
    MAX_CHARS_PER_BATCH = 300_000

    def __init__(self):
        self._nlp_cache = {}

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

    def _split_text_in_batches(self, text: str, max_chars: int) -> list[tuple[int, str]]:
        if len(text) <= max_chars:
            return [(0, text)]

        batches: list[tuple[int, str]] = []
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
