from __future__ import annotations

from typing import Dict, Type

from src.retrieval.base import BaseRetriever
from src.retrieval.faiss_retriever import FAISSRetriever
from src.retrieval.numpy_retriever import NumpyRetriever


class RetrieverFactory:
    _registry: Dict[str, Type[BaseRetriever]] = {
        "faiss": FAISSRetriever,
        "numpy": NumpyRetriever,
    }

    @classmethod
    def register(cls, name: str, retriever_cls: Type[BaseRetriever]) -> None:
        cls._registry[name] = retriever_cls

    @classmethod
    def create(cls, name: str) -> BaseRetriever:
        if name not in cls._registry:
            available = ", ".join(sorted(cls._registry.keys()))
            raise ValueError(f"Unknown retriever: '{name}'. Available: {available}")
        return cls._registry[name]()