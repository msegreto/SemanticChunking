from __future__ import annotations

from typing import Dict, Type

from src.splitting.base import BaseSplitter
from src.splitting.proposition import PropositionSplitter
from src.splitting.sentence import SentenceSplitter


class SplitterFactory:
    _registry: Dict[str, Type[BaseSplitter]] = {
        "sentence": SentenceSplitter,
        "proposition": PropositionSplitter,
    }

    @classmethod
    def register(cls, name: str, splitter_cls: Type[BaseSplitter]) -> None:
        cls._registry[name] = splitter_cls

    @classmethod
    def create(cls, name: str) -> BaseSplitter:
        if name not in cls._registry:
            available = ", ".join(sorted(cls._registry.keys()))
            raise ValueError(f"Unknown splitter: '{name}'. Available: {available}")
        return cls._registry[name]()