from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseSplitter(ABC):
    def resolve_output_path(self, config: dict) -> Path | None:
        output_path = config.get("output_path")
        return Path(output_path) if output_path else None

    @abstractmethod
    def build_streaming_components(self, config: dict) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def split_document_streaming(
        self,
        *,
        doc_id: str,
        text: str,
        unit_id_start: int,
        nlp: Any,
        max_chars_per_batch: int,
    ) -> tuple[list[dict[str, Any]], int]:
        raise NotImplementedError
