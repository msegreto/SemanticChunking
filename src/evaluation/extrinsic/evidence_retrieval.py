from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Sequence

from src.embeddings.factory import EmbedderFactory
from src.retrieval.factory import RetrieverFactory

from .base import BaseExtrinsicEvaluator
from .common import build_base_row, build_run_context
from .io import (
    load_evidences,
    load_index_metadata,
    load_queries,
    resolve_evidence_path,
    resolve_queries_path,
)


class EvidenceRetrievalEvaluator(BaseExtrinsicEvaluator):
    @property
    def task_name(self) -> str:
        return "evidence_retrieval"

    def evaluate(
        self,
        *,
        config: dict[str, Any],
        index_path: Path,
        index_metadata_path: Path,
        ks: Sequence[int],
    ) -> list[dict[str, Any]]:
        try:
            evidence_file = resolve_evidence_path(config)
        except Exception:
            evidence_file = None

        if not ks:
            return self._build_rows(
                config=config,
                ks=[None],
                index_path=index_path,
                index_metadata_path=index_metadata_path,
                status="skipped",
                details="Task skipped: empty ks.",
                evidence_path=None,
            )

        resolved_ks = sorted({int(k) for k in ks})

        if evidence_file is None:
            return self._build_rows(
                config=config,
                ks=resolved_ks,
                index_path=index_path,
                index_metadata_path=index_metadata_path,
                status="skipped",
                details="Task skipped: missing evidence_path in YAML.",
                evidence_path=None,
            )

        if not evidence_file.exists():
            return self._build_rows(
                config=config,
                ks=resolved_ks,
                index_path=index_path,
                index_metadata_path=index_metadata_path,
                status="skipped",
                details=f"Task skipped: evidence annotations file not found at {evidence_file}.",
                evidence_path=str(evidence_file),
            )

        queries_path = resolve_queries_path(config)
        queries = load_queries(queries_path)
        evidences = load_evidences(evidence_file)
        items = load_index_metadata(index_metadata_path)["items"]

        ordered_qids = [qid for qid in queries.keys() if qid in evidences]
        if not ordered_qids:
            return self._build_rows(
                config=config,
                ks=resolved_ks,
                index_path=index_path,
                index_metadata_path=index_metadata_path,
                status="skipped",
                details="Task skipped: no overlapping query ids between queries and evidences.",
                evidence_path=str(evidence_file),
            )

        retrieval_cfg = config["retrieval"]
        embedding_cfg = config["embedding"]
        embedder = self._load_embedder(embedding_cfg)
        retriever = self._load_retriever(retrieval_cfg)
        retriever.load_index(str(index_path), str(index_metadata_path))

        query_texts = [queries[qid] for qid in ordered_qids]
        query_embeddings = self._encode_queries(embedder, query_texts, embedding_cfg)
        search_output = retriever.search(query_embeddings, top_k=max(resolved_ks))
        all_indices = search_output["indices"]

        metrics: dict[int, dict[str, float]] = {}
        evaluable_count = 0
        for k in resolved_ks:
            precision_sum = 0.0
            recall_sum = 0.0
            f1_sum = 0.0
            count = 0

            for row_idx, qid in enumerate(ordered_qids):
                annotations = evidences.get(qid, [])
                gold_texts = [
                    ann.get("evidence_text", "").strip()
                    for ann in annotations
                    if ann.get("evidence_text", "").strip()
                ]
                if not gold_texts:
                    continue

                top_chunk_indices = all_indices[row_idx].tolist()[:k]
                retrieved_sentences = self._collect_retrieved_sentences(top_chunk_indices, items)
                p, r, f1 = self._sentence_level_prf(
                    retrieved_sentences=retrieved_sentences,
                    gold_evidence_texts=gold_texts,
                )

                precision_sum += p
                recall_sum += r
                f1_sum += f1
                count += 1

            metrics[k] = {
                "precision": (precision_sum / count) if count > 0 else 0.0,
                "recall": (recall_sum / count) if count > 0 else 0.0,
                "f1": (f1_sum / count) if count > 0 else 0.0,
                "num_evaluable_queries": count,
            }
            evaluable_count = max(evaluable_count, count)

        if evaluable_count == 0:
            return self._build_rows(
                config=config,
                ks=resolved_ks,
                index_path=index_path,
                index_metadata_path=index_metadata_path,
                status="skipped",
                details="Task skipped: no non-empty evidence annotations found.",
                evidence_path=str(evidence_file),
            )

        return self._build_scored_rows(
            config=config,
            ks=resolved_ks,
            metrics=metrics,
            num_queries=len(ordered_qids),
            num_evaluable_queries=evaluable_count,
            queries_path=str(queries_path),
            index_path=index_path,
            index_metadata_path=index_metadata_path,
            evidence_path=str(evidence_file),
        )

    @staticmethod
    def _collect_retrieved_sentences(
        chunk_indices: list[int],
        items: list[dict[str, Any]],
    ) -> list[str]:
        sentences: list[str] = []
        for chunk_idx in chunk_indices:
            if chunk_idx < 0 or chunk_idx >= len(items):
                continue
            item = items[chunk_idx]
            metadata = item.get("metadata", {}) if isinstance(item, dict) else {}
            chunk_sentences = metadata.get("sentences", []) if isinstance(metadata, dict) else []

            if isinstance(chunk_sentences, list) and chunk_sentences:
                for sentence in chunk_sentences:
                    if isinstance(sentence, str) and sentence.strip():
                        sentences.append(sentence.strip())
                continue

            text = item.get("text", "") if isinstance(item, dict) else ""
            if isinstance(text, str) and text.strip():
                sentences.extend(EvidenceRetrievalEvaluator._split_into_sentences(text))
        return sentences

    @staticmethod
    def _split_into_sentences(text: str) -> list[str]:
        pieces = re.split(r"(?<=[.!?])\s+|\n+", text)
        return [piece.strip() for piece in pieces if piece.strip()]

    @classmethod
    def _normalize_sentence(cls, text: str) -> str:
        lowered = text.lower()
        lowered = re.sub(r"\s+", " ", lowered).strip()
        return lowered

    @classmethod
    def _sentence_matches(cls, retrieved: str, gold: str) -> bool:
        r = cls._normalize_sentence(retrieved)
        g = cls._normalize_sentence(gold)
        if not r or not g:
            return False
        # Evidence snippets in RAGBench can be shorter than a full sentence.
        return (r == g) or (g in r) or (r in g)

    @classmethod
    def _sentence_level_prf(
        cls,
        *,
        retrieved_sentences: list[str],
        gold_evidence_texts: list[str],
    ) -> tuple[float, float, float]:
        gold_sentences: list[str] = []
        for text in gold_evidence_texts:
            gold_sentences.extend(cls._split_into_sentences(text))

        retrieved_norm = [cls._normalize_sentence(s) for s in retrieved_sentences if cls._normalize_sentence(s)]
        gold_norm = [cls._normalize_sentence(s) for s in gold_sentences if cls._normalize_sentence(s)]

        if not gold_norm:
            return 0.0, 0.0, 0.0
        if not retrieved_norm:
            return 0.0, 0.0, 0.0

        # Presence-based counting, aligned with "evidence sentences present in top-k chunks".
        retrieved_unique = list(dict.fromkeys(retrieved_norm))
        gold_unique = list(dict.fromkeys(gold_norm))

        matched_retrieved = 0
        matched_gold: set[int] = set()
        for retrieved in retrieved_unique:
            found = False
            for gold_idx, gold in enumerate(gold_unique):
                if cls._sentence_matches(retrieved, gold):
                    found = True
                    matched_gold.add(gold_idx)
            if found:
                matched_retrieved += 1

        precision = (
            float(matched_retrieved) / float(len(retrieved_unique))
            if retrieved_unique
            else 0.0
        )
        recall = float(len(matched_gold)) / float(len(gold_unique)) if gold_unique else 0.0
        if precision + recall == 0.0:
            return 0.0, 0.0, 0.0
        return precision, recall, (2.0 * precision * recall) / (precision + recall)

    @staticmethod
    def _load_embedder(embedding_cfg: dict[str, Any]):
        embedder_name = embedding_cfg.get("name")
        if not isinstance(embedder_name, str) or not embedder_name.strip():
            raise ValueError("Embedding config requires a non-empty 'name' for extrinsic evaluation.")
        return EmbedderFactory.create(embedder_name.strip())

    @staticmethod
    def _load_retriever(retrieval_cfg: dict[str, Any]):
        retriever_name = retrieval_cfg.get("name", "")
        return RetrieverFactory.create(str(retriever_name).strip().lower())

    @staticmethod
    def _encode_queries(embedder, query_texts: list[str], embedding_cfg: dict[str, Any]):
        query_embeddings, _ = embedder.encode_texts(query_texts, embedding_cfg)
        return query_embeddings

    def _build_scored_rows(
        self,
        *,
        config: dict[str, Any],
        ks: Sequence[int],
        metrics: dict[int, dict[str, float]],
        num_queries: int,
        num_evaluable_queries: int,
        queries_path: str,
        index_path: Path,
        index_metadata_path: Path,
        evidence_path: str | None,
    ) -> list[dict[str, Any]]:
        context = build_run_context(
            config=config,
            task_name=self.task_name,
            default_experiment_name="evidence-retrieval",
        )
        rows: list[dict[str, Any]] = []
        for k in ks:
            row = build_base_row(
                context=context,
                k=k,
                index_path=index_path,
                index_metadata_path=index_metadata_path,
            )
            row.update(
                {
                    "precision": metrics[k]["precision"],
                    "recall": metrics[k]["recall"],
                    "f1": metrics[k]["f1"],
                    "num_queries": num_queries,
                    "num_evaluable_queries": num_evaluable_queries,
                    "queries_path": queries_path,
                    "evidence_path": evidence_path,
                }
            )
            rows.append(row)
        return rows

    def _build_rows(
        self,
        *,
        config: dict[str, Any],
        ks: Sequence[int | None],
        index_path: Path,
        index_metadata_path: Path,
        status: str,
        details: str,
        evidence_path: str | None,
    ) -> list[dict[str, Any]]:
        context = build_run_context(
            config=config,
            task_name=self.task_name,
            default_experiment_name="evidence-retrieval",
        )
        rows: list[dict[str, Any]] = []
        for k in ks:
            row = build_base_row(
                context=context,
                k=k,
                index_path=index_path,
                index_metadata_path=index_metadata_path,
            )
            row.update(
                {
                    "status": status,
                    "details": details,
                    "evidence_path": evidence_path,
                }
            )
            rows.append(row)
        return rows
