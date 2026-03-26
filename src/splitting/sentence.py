from __future__ import annotations

from pathlib import Path
from typing import Any

import spacy

class SentenceSplitter:
    MAX_CHARS_PER_BATCH = 300_000

    def __init__(self):
        self._nlp_cache = {}

    @staticmethod
    def resolve_output_path(config: dict) -> Path | None:
        output_path = config.get("output_path")
        return Path(output_path) if output_path else None

    def build_streaming_components(self, config: dict) -> dict[str, Any]:
        model_name = config.get("model", "en_core_web_sm")
        allow_blank_fallback = bool(config.get("allow_blank_fallback", False))
        nlp = self._load_spacy_model(model_name, allow_blank_fallback=allow_blank_fallback)
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
        docno: str,
        text: str,
        unitno_start: int,
        nlp: Any,
        max_chars_per_batch: int,
    ) -> tuple[list[dict[str, Any]], int]:
        if not text or not str(text).strip():
            return [], unitno_start

        units: list[dict[str, Any]] = []
        next_unitno = unitno_start
        position = 0
        for batch_start, batch_text in self._split_text_in_batches(text, max_chars_per_batch):
            parsed = nlp(batch_text)
            for sent in parsed.sents:
                sent_text = sent.text.strip()
                if not sent_text:
                    continue
                units.append(
                    {
                        "unitno": next_unitno,
                        "parent_docno": docno,
                        "position": position,
                        "text": sent_text,
                        "char_start": batch_start + sent.start_char,
                        "char_end": batch_start + sent.end_char,
                    }
                )
                next_unitno += 1
                position += 1

        return units, next_unitno

    def _load_spacy_model(self, model_name: str, *, allow_blank_fallback: bool):

        if model_name in self._nlp_cache:
            return self._nlp_cache[model_name]

        try:
            nlp = spacy.load(model_name)

        except Exception as exc:
            if not allow_blank_fallback:
                raise RuntimeError(
                    f"Unable to load spaCy model '{model_name}'. "
                    "Install it or set split.allow_blank_fallback=true to opt into a blank English sentencizer."
                ) from exc
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
