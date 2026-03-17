from __future__ import annotations

from typing import Any

from src.splitting.base import BaseSplitter


class PropositionSplitter(BaseSplitter):
    def build_streaming_components(self, config: dict) -> dict[str, Any]:
        raise ValueError("Streaming mode currently supports only split.type='sentence'.")

    def split_document_streaming(
        self,
        *,
        doc_id: str,
        text: str,
        unit_id_start: int,
        nlp: Any,
        max_chars_per_batch: int,
    ) -> tuple[list[dict[str, Any]], int]:
        raise ValueError("Streaming mode currently supports only split.type='sentence'.")
