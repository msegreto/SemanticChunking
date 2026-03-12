from __future__ import annotations

from typing import Dict, Type

from src.evaluation.intrinsic.base import BaseIntrinsicEvaluator
from src.evaluation.intrinsic.dummy import DummyIntrinsicEvaluator


class IntrinsicEvaluatorFactory:
    _registry: Dict[str, Type[BaseIntrinsicEvaluator]] = {
        "default": DummyIntrinsicEvaluator,
    }

    @classmethod
    def register(cls, name: str, evaluator_cls: Type[BaseIntrinsicEvaluator]) -> None:
        cls._registry[name] = evaluator_cls

    @classmethod
    def create(cls, name: str) -> BaseIntrinsicEvaluator:
        if name not in cls._registry:
            available = ", ".join(sorted(cls._registry.keys()))
            raise ValueError(f"Unknown intrinsic evaluator: '{name}'. Available: {available}")
        return cls._registry[name]()