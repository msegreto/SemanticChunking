from __future__ import annotations

import math
import random
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
        query_subsets = self._resolve_query_subsets(config=config, ordered_qids=ordered_qids)

        embedder = self._load_embedder(embedding_cfg)
        retriever = self._load_retriever(retrieval_cfg)
        retriever.load_index(str(index_path), str(index_metadata_path))

        query_texts = [queries[qid] for qid in ordered_qids]
        query_embeddings = self._encode_queries(embedder, query_texts, embedding_cfg)
        search_output = retriever.search(query_embeddings, top_k=max(ks))
        all_indices = search_output["indices"]
        all_scores = search_output["scores"]

        ranked_doc_ids_by_qid: dict[str, list[str]] = {}
        ranked_doc_scores_by_qid: dict[str, list[tuple[str, float]]] = {}

        for row_idx, qid in enumerate(ordered_qids):
            ranked_doc_scores = self._aggregate_chunk_hits_to_documents(
                chunk_indices=all_indices[row_idx].tolist(),
                chunk_scores=all_scores[row_idx].tolist(),
                items=items,
            )
            ranked_doc_ids_by_qid[qid] = [doc_id for doc_id, _ in ranked_doc_scores]
            ranked_doc_scores_by_qid[qid] = ranked_doc_scores

        rows: list[dict[str, Any]] = []
        for subset in query_subsets:
            subset_qids = subset["ordered_qids"]
            subset_qrels = {qid: qrels[qid] for qid in subset_qids}
            subset_ranked_doc_ids_by_qid = {qid: ranked_doc_ids_by_qid[qid] for qid in subset_qids}
            subset_ir_qrels = {
                qid: {doc_id: 1 for doc_id in subset_qrels[qid]}
                for qid in subset_qids
            }
            subset_ir_run = {
                qid: {doc_id: float(score) for doc_id, score in ranked_doc_scores_by_qid[qid]}
                for qid in subset_qids
            }
            metrics = self._compute_metrics(
                ks=ks,
                ir_qrels=subset_ir_qrels,
                ir_run=subset_ir_run,
                ranked_doc_ids_by_qid=subset_ranked_doc_ids_by_qid,
                qrels=subset_qrels,
            )
            rows.extend(
                self._build_rows(
                    ks=ks,
                    metrics=metrics,
                    run_context=run_context,
                    num_queries=len(subset_qids),
                    queries_path=queries_path,
                    qrels_path=qrels_path,
                    index_path=index_path,
                    index_metadata_path=index_metadata_path,
                    query_subset_label=str(subset["label"]),
                    query_subset_size_requested=subset["size_requested"],
                    query_subset_seed=subset["seed"],
                )
            )
        return rows

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
        ndcg_measures_by_k = {k: ir_measures.nDCG @ k for k in ks}

        aggregate_results = ir_measures.calc_aggregate(
            list(ndcg_measures_by_k.values()),
            ir_qrels,
            ir_run,
        )

        metrics: dict[int, dict[str, float]] = {}
        for k in ks:
            macro_precision, macro_recall, macro_f1 = self._macro_prf_at_k(
                ranked_doc_ids_by_qid=ranked_doc_ids_by_qid,
                qrels=qrels,
                k=k,
            )
            micro_precision, micro_recall, micro_f1 = self._micro_prf_at_k(
                ranked_doc_ids_by_qid=ranked_doc_ids_by_qid,
                qrels=qrels,
                k=k,
            )
            metrics[k] = {
                "macro_precision": macro_precision,
                "macro_recall": macro_recall,
                "macro_f1": macro_f1,
                "micro_precision": micro_precision,
                "micro_recall": micro_recall,
                "micro_f1": micro_f1,
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
        query_subset_label: str,
        query_subset_size_requested: int | None,
        query_subset_seed: int | None,
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
                    # Legacy columns mirror macro metrics for compatibility.
                    "precision": metrics[k]["macro_precision"],
                    "recall": metrics[k]["macro_recall"],
                    "f1": metrics[k]["macro_f1"],
                    "macro_precision": metrics[k]["macro_precision"],
                    "macro_recall": metrics[k]["macro_recall"],
                    "macro_f1": metrics[k]["macro_f1"],
                    "micro_precision": metrics[k]["micro_precision"],
                    "micro_recall": metrics[k]["micro_recall"],
                    "micro_f1": metrics[k]["micro_f1"],
                    "dcg": metrics[k]["dcg"],
                    "ndcg": metrics[k]["ndcg"],
                    "num_queries": num_queries,
                    "queries_path": str(queries_path),
                    "qrels_path": str(qrels_path),
                    "query_subset": query_subset_label,
                    "query_subset_size_requested": query_subset_size_requested,
                    "query_subset_seed": query_subset_seed,
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
    def _macro_prf_at_k(
        *,
        ranked_doc_ids_by_qid: dict[str, list[str]],
        qrels: dict[str, set[str]],
        k: int,
    ) -> tuple[float, float, float]:
        if not ranked_doc_ids_by_qid:
            return 0.0, 0.0, 0.0

        precisions: list[float] = []
        recalls: list[float] = []
        f1s: list[float] = []
        for qid, ranked_doc_ids in ranked_doc_ids_by_qid.items():
            gold_doc_ids = qrels.get(qid, set())
            retrieved_at_k = ranked_doc_ids[:k]
            true_positives = float(sum(1 for doc_id in retrieved_at_k if doc_id in gold_doc_ids))
            retrieved_count = float(len(retrieved_at_k))
            precision = true_positives / retrieved_count if retrieved_count > 0.0 else 0.0
            recall = (
                true_positives / float(len(gold_doc_ids))
                if gold_doc_ids
                else 0.0
            )
            precisions.append(precision)
            recalls.append(recall)
            if precision + recall == 0.0:
                f1s.append(0.0)
            else:
                f1s.append((2.0 * precision * recall) / (precision + recall))
        return (
            float(sum(precisions) / len(precisions)) if precisions else 0.0,
            float(sum(recalls) / len(recalls)) if recalls else 0.0,
            float(sum(f1s) / len(f1s)) if f1s else 0.0,
        )

    @staticmethod
    def _micro_prf_at_k(
        *,
        ranked_doc_ids_by_qid: dict[str, list[str]],
        qrels: dict[str, set[str]],
        k: int,
    ) -> tuple[float, float, float]:
        if not ranked_doc_ids_by_qid:
            return 0.0, 0.0, 0.0

        total_tp = 0.0
        total_retrieved = 0.0
        total_relevant = 0.0

        for qid, ranked_doc_ids in ranked_doc_ids_by_qid.items():
            gold_doc_ids = qrels.get(qid, set())
            retrieved_at_k = ranked_doc_ids[:k]
            tp = float(sum(1 for doc_id in retrieved_at_k if doc_id in gold_doc_ids))
            total_tp += tp
            total_retrieved += float(len(retrieved_at_k))
            total_relevant += float(len(gold_doc_ids))

        precision = total_tp / total_retrieved if total_retrieved > 0.0 else 0.0
        recall = total_tp / total_relevant if total_relevant > 0.0 else 0.0
        f1 = (2.0 * precision * recall) / (precision + recall) if precision + recall > 0.0 else 0.0
        return precision, recall, f1

    @staticmethod
    def _resolve_query_subsets(
        *,
        config: dict[str, Any],
        ordered_qids: list[str],
    ) -> list[dict[str, Any]]:
        task_cfg = (
            config.get("evaluation", {})
            .get("extrinsic_tasks", {})
            .get("document_retrieval", {})
        )
        seed_raw = task_cfg.get("query_subset_seed", 42)
        seed = int(seed_raw) if isinstance(seed_raw, int) else 42
        sample_size_raw = task_cfg.get("query_subset_size")
        include_full = bool(task_cfg.get("include_full_query_set", True))

        subsets: list[dict[str, Any]] = []
        seen_labels: set[str] = set()

        def _push_subset(
            *,
            label: str,
            selected_qids: list[str],
            size_requested: int | None,
            subset_seed: int | None,
        ) -> None:
            if label in seen_labels:
                return
            seen_labels.add(label)
            subsets.append(
                {
                    "label": label,
                    "ordered_qids": selected_qids,
                    "size_requested": size_requested,
                    "seed": subset_seed,
                }
            )

        if include_full:
            _push_subset(
                label="all",
                selected_qids=ordered_qids,
                size_requested=None,
                subset_seed=None,
            )

        if isinstance(sample_size_raw, int) and sample_size_raw > 0:
            target_size = min(sample_size_raw, len(ordered_qids))
            rng = random.Random(seed)
            sampled = set(rng.sample(ordered_qids, target_size))
            sampled_qids = [qid for qid in ordered_qids if qid in sampled]
            _push_subset(
                label=f"sample_{target_size}",
                selected_qids=sampled_qids,
                size_requested=sample_size_raw,
                subset_seed=seed,
            )

        if not subsets:
            _push_subset(
                label="all",
                selected_qids=ordered_qids,
                size_requested=None,
                subset_seed=None,
            )
        return subsets

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
