from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import ir_measures
from src.embeddings.factory import EmbedderFactory
from src.retrieval.factory import RetrieverFactory

from .base import BaseExtrinsicEvaluator
from .common import ExtrinsicRunContext, build_base_row, build_run_context
from .io import (
    load_index_metadata,
    load_qrels,
    load_queries,
    resolve_qrels_path,
    resolve_queries_path,
)


class DocumentRetrievalEvaluator(BaseExtrinsicEvaluator):
    @property
    def task_name(self) -> str:
        return "document_retrieval"

    def evaluate(
        self,
        *,
        config: dict[str, Any],
        index_path: Path,
        index_metadata_path: Path,
        ks: Sequence[int],
    ) -> list[dict[str, Any]]:
        ks = sorted({int(k) for k in ks})
        if not ks:
            raise ValueError("Parameter 'ks' must contain at least one value.")

        retrieval_cfg = config["retrieval"]
        embedding_cfg = config["embedding"]
        run_context = build_run_context(
            config=config,
            task_name=self.task_name,
            default_experiment_name="document-retrieval",
        )

        queries_path, qrels_path, queries, qrels, items = self._load_eval_inputs(
            config=config,
            index_metadata_path=index_metadata_path,
        )
        ordered_qids = self._resolve_ordered_qids(
            queries=queries,
            qrels=qrels,
        )

        embedder = self._load_embedder(embedding_cfg)
        retriever = self._load_retriever(retrieval_cfg)
        retriever.load_index(str(index_path), str(index_metadata_path))

        query_texts = [queries[qid] for qid in ordered_qids]
        query_embeddings = self._encode_queries(embedder, query_texts, embedding_cfg)
        search_output = retriever.search(query_embeddings, top_k=max(ks))
        all_indices = search_output["indices"]
        all_scores = search_output["scores"]

        ir_qrels: dict[str, dict[str, int]] = {}
        ir_run: dict[str, dict[str, float]] = {}
        ranked_doc_ids_by_qid: dict[str, list[str]] = {}

        for row_idx, qid in enumerate(ordered_qids):
            gold_doc_ids = qrels[qid]
            ranked_doc_scores = self._aggregate_chunk_hits_to_documents(
                chunk_indices=all_indices[row_idx].tolist(),
                chunk_scores=all_scores[row_idx].tolist(),
                items=items,
            )
            ranked_doc_ids_by_qid[qid] = [doc_id for doc_id, _ in ranked_doc_scores]

            ir_qrels[qid] = {doc_id: 1 for doc_id in gold_doc_ids}
            ir_run[qid] = {doc_id: float(score) for doc_id, score in ranked_doc_scores}
        metrics = self._compute_metrics(
            ks=ks,
            ir_qrels=ir_qrels,
            ir_run=ir_run,
            ranked_doc_ids_by_qid=ranked_doc_ids_by_qid,
            qrels=qrels,
        )
        return self._build_rows(
            ks=ks,
            metrics=metrics,
            run_context=run_context,
            num_queries=len(ordered_qids),
            queries_path=queries_path,
            qrels_path=qrels_path,
            index_path=index_path,
            index_metadata_path=index_metadata_path,
        )

    def _load_eval_inputs(
        self,
        *,
        config: dict[str, Any],
        index_metadata_path: Path,
    ) -> tuple[Path, Path, dict[str, str], dict[str, set[str]], list[dict[str, Any]]]:
        queries_path = resolve_queries_path(config)
        qrels_path = resolve_qrels_path(config)
        queries = load_queries(queries_path)
        qrels = load_qrels(qrels_path)
        items = load_index_metadata(index_metadata_path)["items"]
        if not items:
            raise ValueError("No indexed items found in metadata.pkl")
        return queries_path, qrels_path, queries, qrels, items

    def _resolve_ordered_qids(
        self,
        *,
        queries: dict[str, str],
        qrels: dict[str, set[str]],
    ) -> list[str]:
        ordered_qids = [qid for qid in queries.keys() if qid in qrels]
        if not ordered_qids:
            raise ValueError("No overlapping query ids between queries.json and qrels")
        return ordered_qids

    def _compute_metrics(
        self,
        *,
        ks: list[int],
        ir_qrels: dict[str, dict[str, int]],
        ir_run: dict[str, dict[str, float]],
        ranked_doc_ids_by_qid: dict[str, list[str]],
        qrels: dict[str, set[str]],
    ) -> dict[int, dict[str, float]]:
        precision_measures_by_k = {k: ir_measures.P @ k for k in ks}
        recall_measures_by_k = {k: ir_measures.R @ k for k in ks}
        ndcg_measures_by_k = {k: ir_measures.nDCG @ k for k in ks}

        aggregate_results = ir_measures.calc_aggregate(
            list(precision_measures_by_k.values())
            + list(recall_measures_by_k.values())
            + list(ndcg_measures_by_k.values()),
            ir_qrels,
            ir_run,
        )

        metrics: dict[int, dict[str, float]] = {}
        for k in ks:
            metrics[k] = {
                "precision": float(aggregate_results.get(precision_measures_by_k[k], 0.0)),
                "recall": float(aggregate_results.get(recall_measures_by_k[k], 0.0)),
                "f1": self._macro_f1_at_k(
                    ranked_doc_ids_by_qid=ranked_doc_ids_by_qid,
                    qrels=qrels,
                    k=k,
                ),
                "dcg": self._macro_dcg_at_k(
                    ranked_doc_ids_by_qid=ranked_doc_ids_by_qid,
                    qrels=qrels,
                    k=k,
                ),
                "ndcg": float(aggregate_results.get(ndcg_measures_by_k[k], 0.0)),
            }
        return metrics

    def _build_rows(
        self,
        *,
        ks: list[int],
        metrics: dict[int, dict[str, float]],
        run_context: ExtrinsicRunContext,
        num_queries: int,
        queries_path: Path,
        qrels_path: Path,
        index_path: Path,
        index_metadata_path: Path,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for k in ks:
            row = build_base_row(
                context=run_context,
                k=k,
                index_path=index_path,
                index_metadata_path=index_metadata_path,
            )
            row.update(
                {
                    "precision": metrics[k]["precision"],
                    "recall": metrics[k]["recall"],
                    "f1": metrics[k]["f1"],
                    "dcg": metrics[k]["dcg"],
                    "ndcg": metrics[k]["ndcg"],
                    "num_queries": num_queries,
                    "queries_path": str(queries_path),
                    "qrels_path": str(qrels_path),
                }
            )
            rows.append(row)
        return rows

    @staticmethod
    def _aggregate_chunk_hits_to_documents(
        chunk_indices: list[int],
        chunk_scores: list[float],
        items: list[dict[str, Any]],
    ) -> list[tuple[str, float]]:
        if len(chunk_indices) != len(chunk_scores):
            raise ValueError("Chunk indices and scores must have the same length.")

        doc_score_max: dict[str, float] = {}
        doc_first_rank: dict[str, int] = {}

        for rank, (chunk_idx, chunk_score) in enumerate(zip(chunk_indices, chunk_scores)):
            if chunk_idx < 0 or chunk_idx >= len(items):
                continue

            item = items[chunk_idx]
            doc_id = item.get("doc_id")
            if doc_id is None:
                raise ValueError("Indexed item does not contain 'doc_id'")

            doc_id = str(doc_id)
            score = float(chunk_score)
            prev = doc_score_max.get(doc_id)
            if prev is None or score > prev:
                doc_score_max[doc_id] = score
            if doc_id not in doc_first_rank:
                doc_first_rank[doc_id] = rank

        ranked = sorted(
            doc_score_max.items(),
            key=lambda x: (-x[1], doc_first_rank.get(x[0], 10**12), x[0]),
        )
        return ranked

    @staticmethod
    def _macro_dcg_at_k(
        *,
        ranked_doc_ids_by_qid: dict[str, list[str]],
        qrels: dict[str, set[str]],
        k: int,
    ) -> float:
        if not ranked_doc_ids_by_qid:
            return 0.0

        def _dcg_for_query(predicted_doc_ids: list[str], gold_doc_ids: set[str], cutoff: int) -> float:
            dcg = 0.0
            for rank, doc_id in enumerate(predicted_doc_ids[:cutoff], start=1):
                rel = 1.0 if doc_id in gold_doc_ids else 0.0
                if rel <= 0.0:
                    continue
                dcg += rel / (math.log2(rank + 1.0))
            return dcg

        values: list[float] = []
        for qid, ranked_doc_ids in ranked_doc_ids_by_qid.items():
            gold_doc_ids = qrels.get(qid, set())
            values.append(_dcg_for_query(ranked_doc_ids, gold_doc_ids, k))

        return float(sum(values) / len(values)) if values else 0.0

    @staticmethod
    def _macro_f1_at_k(
        *,
        ranked_doc_ids_by_qid: dict[str, list[str]],
        qrels: dict[str, set[str]],
        k: int,
    ) -> float:
        if not ranked_doc_ids_by_qid:
            return 0.0

        values: list[float] = []
        for qid, ranked_doc_ids in ranked_doc_ids_by_qid.items():
            gold_doc_ids = qrels.get(qid, set())
            retrieved_at_k = ranked_doc_ids[:k]
            true_positives = sum(1 for doc_id in retrieved_at_k if doc_id in gold_doc_ids)
            precision = float(true_positives) / float(k) if k > 0 else 0.0
            recall = (
                float(true_positives) / float(len(gold_doc_ids))
                if gold_doc_ids
                else 0.0
            )
            if precision + recall == 0.0:
                values.append(0.0)
            else:
                values.append((2.0 * precision * recall) / (precision + recall))
        return float(sum(values) / len(values)) if values else 0.0

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
