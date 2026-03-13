from __future__ import annotations

from typing import Any

import numpy as np


SUPPORTED_SIMILARITY_METRICS = {"cosine"}
SUPPORTED_BREAKPOINT_THRESHOLD_TYPES = {
    "percentile",
    "standard_deviation",
    "interquartile",
    "gradient",
}


def validate_common_semantic_config(config: dict[str, Any]) -> dict[str, Any]:
    embedding_model = config.get("embedding_model")
    if not isinstance(embedding_model, str) or not embedding_model.strip():
        raise ValueError(
            "Semantic chunking config requires 'embedding_model' as a non-empty string."
        )

    similarity_metric = config.get("similarity_metric", "cosine")
    if not isinstance(similarity_metric, str) or not similarity_metric.strip():
        raise ValueError(
            "Semantic chunking config requires 'similarity_metric' as a non-empty string."
        )

    similarity_metric = similarity_metric.strip().lower()
    if similarity_metric not in SUPPORTED_SIMILARITY_METRICS:
        available = ", ".join(sorted(SUPPORTED_SIMILARITY_METRICS))
        raise ValueError(
            "Semantic chunking currently supports only "
            f"{available}. Got '{similarity_metric}'."
        )

    batch_size = config.get("embedding_batch_size", 32)
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError(
            "Semantic chunking config requires 'embedding_batch_size' as a positive integer."
        )

    prepared = dict(config)
    prepared["embedding_model"] = embedding_model.strip()
    prepared["similarity_metric"] = similarity_metric
    prepared["embedding_batch_size"] = batch_size
    return prepared


def build_common_semantic_metadata(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "embedding_model": config["embedding_model"],
        "similarity_metric": config["similarity_metric"],
        "embedding_batch_size": config["embedding_batch_size"],
    }


def sort_units_by_sentence_idx(units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(units, key=lambda item: item["sentence_idx"])


def l2_normalize_embeddings(embeddings: Any) -> np.ndarray:
    matrix = np.asarray(embeddings, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D embedding matrix, got shape {matrix.shape}.")

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return matrix / norms


def consecutive_cosine_distances(embeddings: Any) -> np.ndarray:
    matrix = l2_normalize_embeddings(embeddings)
    if matrix.shape[0] < 2:
        return np.asarray([], dtype=float)

    similarities = np.sum(matrix[:-1] * matrix[1:], axis=1)
    similarities = np.clip(similarities, -1.0, 1.0)
    return 1.0 - similarities


def gradient_scores(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return array
    if array.size == 1:
        return np.asarray([0.0], dtype=float)
    return np.gradient(array)
