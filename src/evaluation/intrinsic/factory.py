from __future__ import annotations

from src.evaluation.intrinsic.default import DefaultIntrinsicEvaluator


class IntrinsicEvaluatorFactory:
    @staticmethod
    def create(name: str):
        normalized = name.lower().strip()

        if normalized in {"default", "moc", "mixture_of_chunking"}:
            return DefaultIntrinsicEvaluator()

        raise ValueError(f"Unknown intrinsic evaluator: {name}")