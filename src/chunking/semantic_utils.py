from __future__ import annotations

from typing import Any

import numpy as np


SUPPORTED_SIMILARITY_METRICS = {"cosine"}
SUPPORTED_BREAKPOINT_THRESHOLD_TYPES = {
    "percentile",
    "standard_deviation",
    "interquartile",
    "gradient",
    "distance",
    "gradient_absolute",
}
SUPPORTED_CLUSTERING_MODES = {
    "single_linkage",
    "dbscan",
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

    show_embedding_progress = config.get("show_embedding_progress", False)
    if not isinstance(show_embedding_progress, bool):
        raise ValueError(
            "Semantic chunking config requires 'show_embedding_progress' as a boolean."
        )

    prepared = dict(config)
    prepared["embedding_model"] = embedding_model.strip()
    prepared["similarity_metric"] = similarity_metric
    prepared["embedding_batch_size"] = batch_size
    prepared["show_embedding_progress"] = show_embedding_progress
    return prepared


def build_common_semantic_metadata(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "embedding_model": config["embedding_model"],
        "similarity_metric": config["similarity_metric"],
        "embedding_batch_size": config["embedding_batch_size"],
        "show_embedding_progress": config["show_embedding_progress"],
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


def clipped_cosine_distance_matrix(embeddings: Any) -> np.ndarray:
    matrix = l2_normalize_embeddings(embeddings)
    if matrix.shape[0] == 0:
        return np.asarray([], dtype=float).reshape(0, 0)

    similarities = matrix @ matrix.T
    similarities = np.clip(similarities, -1.0, 1.0)
    similarities = np.maximum(similarities, 0.0)
    distances = 1.0 - similarities
    np.fill_diagonal(distances, 0.0)
    return distances


def positional_distance_matrix(num_items: int) -> np.ndarray:
    if num_items < 0:
        raise ValueError(f"num_items must be >= 0, got {num_items}.")
    if num_items == 0:
        return np.asarray([], dtype=float).reshape(0, 0)

    indices = np.arange(num_items, dtype=float)
    distances = np.abs(indices[:, None] - indices[None, :]) / float(num_items)
    np.fill_diagonal(distances, 0.0)
    return distances


def joint_position_semantic_distance_matrix(
    embeddings: Any,
    *,
    positional_weight: float,
) -> np.ndarray:
    if positional_weight < 0.0 or positional_weight > 1.0:
        raise ValueError(
            f"positional_weight must be in [0, 1], got {positional_weight}."
        )

    semantic_distances = clipped_cosine_distance_matrix(embeddings)
    num_items = semantic_distances.shape[0]
    positional_distances = positional_distance_matrix(num_items)

    joint = (
        positional_weight * positional_distances
        + (1.0 - positional_weight) * semantic_distances
    )
    np.fill_diagonal(joint, 0.0)
    return joint
