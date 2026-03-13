from __future__ import annotations

from typing import Any

from src.chunking.base import BaseChunker


class SemanticChunker(BaseChunker):
    @property
    def chunking_type(self) -> str:
        return "semantic"

    def chunk(self, routed_output: Any, config: dict) -> Any:
        print(f"[CHUNKING] semantic chunking alias with config={config}")
        return super().chunk(routed_output, config)

    def validate_strategy_config(self, config: dict[str, Any]) -> dict[str, Any]:
        raise ValueError(
            "Chunking type 'semantic' is ambiguous. "
            "Use 'semantic_breakpoint' or 'semantic_clustering' to match the paper methods."
        )

    def chunk_grouped_units(
        self,
        grouped_units: dict[str, list[dict[str, Any]]],
        config: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        raise NotImplementedError

    def saved_payload_matches(
        self,
        metadata: Any,
        doc_id: str,
        config: dict[str, Any],
    ) -> bool:
        return False

    def build_saved_payload_metadata(
        self,
        doc_id: str,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        raise NotImplementedError

    def build_strategy_config_metadata(self, config: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError
