from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExtrinsicRunContext:
    dataset: str
    chunking: str
    routing: str
    experiment: str
    task: str
    retriever: str
    embedder: str


def build_run_context(
    *,
    config: dict[str, Any],
    task_name: str,
    default_experiment_name: str,
) -> ExtrinsicRunContext:
    dataset_name = config.get("dataset", {}).get("name", "unknown-dataset")
    chunking_name = config.get("chunking", {}).get("type", "unknown-chunking")
    router_cfg = config.get("router", {})
    routing_name = router_cfg.get("name", "no-routing") if router_cfg.get("enabled", False) else "no-routing"
    experiment_name = (
        config.get("experiment_name")
        or config.get("experiment", {}).get("name")
        or default_experiment_name
    )
    retrieval_cfg = config.get("retrieval", {})
    retriever_name = (
        retrieval_cfg.get("backend")
        or retrieval_cfg.get("wmodel")
        or "unknown-retrieval"
    )
    embedder_name = config.get("embedding", {}).get("name", "unknown-embedder")
    return ExtrinsicRunContext(
        dataset=dataset_name,
        chunking=chunking_name,
        routing=routing_name,
        experiment=experiment_name,
        task=task_name,
        retriever=retriever_name,
        embedder=embedder_name,
    )


def build_base_row(
    *,
    context: ExtrinsicRunContext,
    k: int | None,
    retrieval_manifest_path: Path | None,
    retrieval_run_path: Path | None,
    retrieval_items_path: Path | None,
) -> dict[str, Any]:
    return {
        "dataset": context.dataset,
        "chunking": context.chunking,
        "routing": context.routing,
        "experiment": context.experiment,
        "task": context.task,
        "retriever": context.retriever,
        "embedder": context.embedder,
        "k": k,
        "precision": None,
        "recall": None,
        "f1": None,
        "macro_precision": None,
        "macro_recall": None,
        "macro_f1": None,
        "micro_precision": None,
        "micro_recall": None,
        "micro_f1": None,
        "dcg": None,
        "ndcg": None,
        "qa_similarity": None,
        "bertscore_f1": None,
        "num_queries": None,
        "num_evaluable_queries": None,
        "queries_path": None,
        "qrels_path": None,
        "query_subset": None,
        "query_subset_size_requested": None,
        "query_subset_seed": None,
        "evidence_path": None,
        "answers_path": None,
        "generation_model": None,
        "retrieval_manifest_path": str(retrieval_manifest_path) if retrieval_manifest_path is not None else None,
        "retrieval_run_path": str(retrieval_run_path) if retrieval_run_path is not None else None,
        "retrieval_items_path": str(retrieval_items_path) if retrieval_items_path is not None else None,
    }
