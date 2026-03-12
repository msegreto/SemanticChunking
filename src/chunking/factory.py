from __future__ import annotations

from typing import Dict, Type

from src.chunking.base import BaseChunker
from src.chunking.fixed import FixedChunker
from src.chunking.semantic import SemanticChunker


class ChunkerFactory:
    _registry: Dict[str, Type[BaseChunker]] = {
        "fixed": FixedChunker,
        "semantic": SemanticChunker,
    }

    @classmethod
    def register(cls, name: str, chunker_cls: Type[BaseChunker]) -> None:
        cls._registry[name] = chunker_cls

    @classmethod
    def create(cls, name: str) -> BaseChunker:
        if name not in cls._registry:
            available = ", ".join(sorted(cls._registry.keys()))
            raise ValueError(f"Unknown chunker: '{name}'. Available: {available}")
        return cls._registry[name]()