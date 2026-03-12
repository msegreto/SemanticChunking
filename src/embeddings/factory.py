from __future__ import annotations

from typing import Dict, Type

from src.embeddings.base import BaseEmbedder
from src.embeddings.bge import BGEEmbedder
from src.embeddings.mpnet import MPNetEmbedder
from src.embeddings.stella import StellaEmbedder


class EmbedderFactory:
    _registry: Dict[str, Type[BaseEmbedder]] = {
        "mpnet": MPNetEmbedder,
        "all-mpnet-base-v2": MPNetEmbedder,
        "sentence-transformers/all-mpnet-base-v2": MPNetEmbedder,

        "bge": BGEEmbedder,
        "bge-large": BGEEmbedder,
        "BAAI/bge-large-en-v1.5": BGEEmbedder,

        "stella": StellaEmbedder,
        "stella_en_1.5B_v5": StellaEmbedder,
        "dunzhang/stella_en_1.5B_v5": StellaEmbedder,
    }

    @classmethod
    def register(cls, name: str, embedder_cls: Type[BaseEmbedder]) -> None:
        cls._registry[name] = embedder_cls

    @classmethod
    def create(cls, name: str) -> BaseEmbedder:
        if name not in cls._registry:
            available = ", ".join(sorted(cls._registry.keys()))
            raise ValueError(f"Unknown embedder: '{name}'. Available: {available}")
        return cls._registry[name]()