from __future__ import annotations

from typing import Any

from src.chunking.base import BaseChunker


class SemanticChunker(BaseChunker):
    def chunk(self, routed_output: Any, config: dict) -> Any:
        print(f"[CHUNKING] semantic chunking with config={config}")
        return {
            "chunks": [],
            "metadata": {
                "chunking_type": "semantic",
                "source_metadata": routed_output.get("metadata", {}),
            },
        }