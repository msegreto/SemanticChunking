from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseIntrinsicEvaluator(ABC):
    @abstractmethod
    def evaluate(self, chunk_output: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute intrinsic chunking metrics from chunk_output.

        Expected return format:
        {
            "evaluator_name": str,
            "metrics": dict,
            "per_document_metrics": list[dict],
            "metadata": dict,
        }
        """
        raise NotImplementedError