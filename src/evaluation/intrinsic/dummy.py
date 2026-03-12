from __future__ import annotations

from typing import Any

from src.evaluation.intrinsic.base import BaseIntrinsicEvaluator


class DummyIntrinsicEvaluator(BaseIntrinsicEvaluator):
    def evaluate(self, chunk_output: Any, config: dict) -> Any:
        print(f"[INTRINSIC] evaluate() with config={config}")
        return {
            "boundary_clarity": None,
            "chunk_stickiness": None,
            "metadata": {"evaluator": config.get("intrinsic_evaluator", "default")},
        }