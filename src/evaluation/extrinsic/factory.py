from __future__ import annotations

from .base import BaseExtrinsicEvaluator
from .document_retrieval import DocumentRetrievalEvaluator


class ExtrinsicEvaluatorFactory:
    @staticmethod
    def create(name: str) -> BaseExtrinsicEvaluator:
        normalized = name.strip().lower().replace("-", "_")
        if normalized in {"document_retrieval", "doc_retrieval", "document"}:
            return DocumentRetrievalEvaluator()
        raise ValueError(f"Unsupported extrinsic evaluator: {name}")