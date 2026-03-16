from __future__ import annotations

from .base import BaseExtrinsicEvaluator
from .answer_generation import AnswerGenerationEvaluator
from .document_retrieval import DocumentRetrievalEvaluator
from .evidence_retrieval import EvidenceRetrievalEvaluator


class ExtrinsicEvaluatorFactory:
    @staticmethod
    def create(name: str) -> BaseExtrinsicEvaluator:
        normalized = name.strip().lower().replace("-", "_")
        if normalized in {"document_retrieval", "doc_retrieval", "document"}:
            return DocumentRetrievalEvaluator()
        if normalized in {"evidence_retrieval", "evidence"}:
            return EvidenceRetrievalEvaluator()
        if normalized in {"answer_generation", "generation", "qa_generation"}:
            return AnswerGenerationEvaluator()
        raise ValueError(f"Unsupported extrinsic evaluator: {name}")
