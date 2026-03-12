from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Sequence

from .base import BaseExtrinsicEvaluator
from .io import (
    load_index_metadata,
    load_qrels,
    load_queries,
    resolve_qrels_path,
    resolve_queries_path,
)
from .metrics import precision_recall_f1


class DocumentRetrievalEvaluator(BaseExtrinsicEvaluator):
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

        dataset_cfg = config["dataset"]
        retrieval_cfg = config["retrieval"]
        embedding_cfg = config["embedding"]

        queries_path = resolve_queries_path(config)
        qrels_path = resolve_qrels_path(config)

        queries = load_queries(queries_path)
        qrels = load_qrels(qrels_path)
        metadata_blob = load_index_metadata(index_metadata_path)
        items = metadata_blob["items"]

        if not items:
            raise ValueError("No indexed items found in metadata.pkl")

        embedder = self._load_embedder(embedding_cfg)
        retriever = self._load_retriever(retrieval_cfg)

        retriever.load_index(str(index_path), str(index_metadata_path))

        ordered_qids = [qid for qid in queries.keys() if qid in qrels]
        if not ordered_qids:
            raise ValueError("No overlapping query ids between queries.json and qrels")

        evaluation_cfg = config.get("evaluation", {})
        task_cfg = evaluation_cfg.get("extrinsic_tasks", {}).get("document_retrieval", {})

        query_sample_size = task_cfg.get("query_sample_size")
        query_sample_seed = int(task_cfg.get("query_sample_seed", 42))

        if query_sample_size is not None:
            query_sample_size = int(query_sample_size)
            if query_sample_size <= 0:
                raise ValueError("query_sample_size must be > 0")

            if len(ordered_qids) > query_sample_size:
                rng = random.Random(query_sample_seed)
                ordered_qids = sorted(rng.sample(ordered_qids, query_sample_size))

        query_texts = [queries[qid] for qid in ordered_qids]
        query_embeddings = self._encode_queries(embedder, query_texts, embedding_cfg)

        max_k = max(ks)
        search_output = retriever.search(query_embeddings, top_k=max_k)
        all_indices = search_output["indices"]

        dataset_name = dataset_cfg.get("name", "unknown-dataset")
        chunking_name = config.get("chunking", {}).get("type", "unknown-chunking")
        router_cfg = config.get("router", {})
        routing_name = router_cfg.get("name", "no-routing") if router_cfg.get("enabled", False) else "no-routing"
        experiment_name = (
            config.get("experiment_name")
            or config.get("experiment", {}).get("name")
            or "document-retrieval"
        )
        retriever_name = retrieval_cfg.get("name", "unknown-retriever")
        embedder_name = embedding_cfg.get("name", "unknown-embedder")

        per_k_scores: dict[int, list[tuple[float, float, float]]] = {k: [] for k in ks}

        for row_idx, qid in enumerate(ordered_qids):
            gold_doc_ids = qrels[qid]
            ranked_doc_ids = self._map_chunk_hits_to_documents(
                chunk_indices=all_indices[row_idx].tolist(),
                items=items,
            )

            for k in ks:
                predicted_doc_ids = ranked_doc_ids[:k]
                tp_count = len(set(predicted_doc_ids) & gold_doc_ids)

                precision, recall, f1 = precision_recall_f1(
                    pred_count=len(predicted_doc_ids),
                    gold_count=len(gold_doc_ids),
                    tp_count=tp_count,
                )
                per_k_scores[k].append((precision, recall, f1))

        rows: list[dict[str, Any]] = []
        for k in ks:
            score_list = per_k_scores[k]

            macro_precision = sum(x[0] for x in score_list) / len(score_list) if score_list else 0.0
            macro_recall = sum(x[1] for x in score_list) / len(score_list) if score_list else 0.0
            macro_f1 = sum(x[2] for x in score_list) / len(score_list) if score_list else 0.0

            rows.append(
                {
                    "dataset": dataset_name,
                    "chunking": chunking_name,
                    "routing": routing_name,
                    "experiment": experiment_name,
                    "task": "document_retrieval",
                    "retriever": retriever_name,
                    "embedder": embedder_name,
                    "k": k,
                    "precision": macro_precision,
                    "recall": macro_recall,
                    "f1": macro_f1,
                    "num_queries": len(ordered_qids),
                    "queries_path": str(queries_path),
                    "qrels_path": str(qrels_path),
                    "index_path": str(index_path),
                    "index_metadata_path": str(index_metadata_path),
                }
            )

        return rows

    @staticmethod
    def _map_chunk_hits_to_documents(chunk_indices: list[int], items: list[dict[str, Any]]) -> list[str]:
        ranked_doc_ids: list[str] = []
        seen_doc_ids: set[str] = set()

        for chunk_idx in chunk_indices:
            if chunk_idx < 0 or chunk_idx >= len(items):
                continue

            item = items[chunk_idx]
            doc_id = item.get("doc_id")
            if doc_id is None:
                raise ValueError("Indexed item does not contain 'doc_id'")

            doc_id = str(doc_id)
            if doc_id in seen_doc_ids:
                continue

            seen_doc_ids.add(doc_id)
            ranked_doc_ids.append(doc_id)

        return ranked_doc_ids

    @staticmethod
    def _load_embedder(embedding_cfg: dict[str, Any]):
        embedder_name = embedding_cfg.get("name", "").lower()

        if "mpnet" in embedder_name:
            from src.embeddings.mpnet import MPNetEmbedder
            return MPNetEmbedder()

        raise ValueError(f"Unsupported embedder for extrinsic evaluation: {embedding_cfg.get('name')}")

    @staticmethod
    def _load_retriever(retrieval_cfg: dict[str, Any]):
        retriever_name = retrieval_cfg.get("name", "").lower()

        if retriever_name == "numpy":
            from src.retrieval.numpy_retriever import NumpyRetriever
            return NumpyRetriever()

        if retriever_name == "faiss":
            from src.retrieval.faiss_retriever import FAISSRetriever
            return FAISSRetriever()

        raise ValueError(f"Unsupported retriever for extrinsic evaluation: {retrieval_cfg.get('name')}")

    @staticmethod
    def _encode_queries(embedder, query_texts: list[str], embedding_cfg: dict[str, Any]):
        model = embedder._load_model()

        return model.encode(
            query_texts,
            batch_size=embedding_cfg.get("batch_size", 32),
            show_progress_bar=embedding_cfg.get("show_progress_bar", False),
            convert_to_numpy=True,
            normalize_embeddings=embedding_cfg.get("normalize_embeddings", True),
        )