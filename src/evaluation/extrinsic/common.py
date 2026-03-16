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
    retriever_name = config.get("retrieval", {}).get("name", "unknown-retriever")
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
    index_path: Path,
    index_metadata_path: Path,
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
        "dcg": None,
        "ndcg": None,
        "num_queries": None,
        "queries_path": None,
        "qrels_path": None,
        "index_path": str(index_path),
        "index_metadata_path": str(index_metadata_path),
    }
