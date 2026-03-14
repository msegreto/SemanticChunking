from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.cluster import DBSCAN
from tqdm.auto import tqdm

from src.chunking.semantic_base import BaseSemanticChunker
from src.chunking.semantic_utils import (
    SUPPORTED_CLUSTERING_MODES,
    joint_position_semantic_distance_matrix,
)


class SemanticClusteringChunker(BaseSemanticChunker):
    @property
    def chunking_type(self) -> str:
        return "semantic_clustering"

    def chunk(self, routed_output: Any, config: dict) -> Any:
        print(f"[CHUNKING] semantic clustering chunking with config={config}")
        return super().chunk(routed_output, config)

    def validate_semantic_method_config(self, config: dict[str, Any]) -> dict[str, Any]:
        clustering_mode = config.get("clustering_mode", "single_linkage")
        if not isinstance(clustering_mode, str) or not clustering_mode.strip():
            raise ValueError(
                "Semantic clustering config requires 'clustering_mode' as a non-empty string."
            )
        clustering_mode = clustering_mode.strip().lower()
        if clustering_mode not in SUPPORTED_CLUSTERING_MODES:
            available = ", ".join(sorted(SUPPORTED_CLUSTERING_MODES))
            raise ValueError(
                "Semantic clustering config requires a supported 'clustering_mode'. "
                f"Available: {available}. Got '{clustering_mode}'."
            )

        lambda_value = config.get("lambda", config.get("lambda_weight", 0.5))
        if isinstance(lambda_value, bool) or not isinstance(lambda_value, (int, float)):
            raise ValueError("Semantic clustering config requires 'lambda' to be numeric.")
        lambda_value = float(lambda_value)
        if not 0.0 <= lambda_value <= 1.0:
            raise ValueError("Semantic clustering config requires 'lambda' in [0, 1].")

        stop_distance_threshold = config.get("stop_distance_threshold", 0.5)
        if isinstance(stop_distance_threshold, bool) or not isinstance(
            stop_distance_threshold,
            (int, float),
        ):
            raise ValueError(
                "Semantic clustering config requires 'stop_distance_threshold' to be numeric."
            )
        stop_distance_threshold = float(stop_distance_threshold)
        if stop_distance_threshold < 0.0:
            raise ValueError(
                "Semantic clustering config requires 'stop_distance_threshold' >= 0."
            )

        allow_non_contiguous_chunks = config.get("allow_non_contiguous_chunks", True)
        if not isinstance(allow_non_contiguous_chunks, bool):
            raise ValueError(
                "Semantic clustering config requires 'allow_non_contiguous_chunks' as a boolean."
            )

        number_of_chunks = config.get("number_of_chunks")
        if clustering_mode == "single_linkage":
            if isinstance(number_of_chunks, bool) or not isinstance(number_of_chunks, int):
                raise ValueError(
                    "Semantic clustering config with 'single_linkage' requires "
                    "'number_of_chunks' as a positive integer."
                )
            if number_of_chunks <= 0:
                raise ValueError(
                    "Semantic clustering config with 'single_linkage' requires "
                    "'number_of_chunks' > 0."
                )
        elif number_of_chunks is not None:
            if isinstance(number_of_chunks, bool) or not isinstance(number_of_chunks, int):
                raise ValueError(
                    "Semantic clustering config requires 'number_of_chunks' to be an integer "
                    "when provided."
                )
            if number_of_chunks <= 0:
                raise ValueError(
                    "Semantic clustering config requires 'number_of_chunks' > 0 when provided."
                )

        dbscan_eps = config.get("dbscan_eps", 0.3)
        if isinstance(dbscan_eps, bool) or not isinstance(dbscan_eps, (int, float)):
            raise ValueError("Semantic clustering config requires 'dbscan_eps' to be numeric.")
        dbscan_eps = float(dbscan_eps)
        if dbscan_eps < 0.0:
            raise ValueError("Semantic clustering config requires 'dbscan_eps' >= 0.")

        dbscan_min_samples = config.get("dbscan_min_samples", 2)
        if isinstance(dbscan_min_samples, bool) or not isinstance(dbscan_min_samples, int):
            raise ValueError(
                "Semantic clustering config requires 'dbscan_min_samples' as an integer."
            )
        if dbscan_min_samples <= 0:
            raise ValueError(
                "Semantic clustering config requires 'dbscan_min_samples' > 0."
            )

        prepared = dict(config)
        prepared["clustering_mode"] = clustering_mode.strip()
        prepared["lambda_weight"] = lambda_value
        prepared["stop_distance_threshold"] = stop_distance_threshold
        prepared["allow_non_contiguous_chunks"] = allow_non_contiguous_chunks
        prepared["number_of_chunks"] = number_of_chunks
        prepared["dbscan_eps"] = dbscan_eps
        prepared["dbscan_min_samples"] = dbscan_min_samples
        prepared["paper_method_family"] = "clustering"
        prepared["implementation_status"] = "implemented"
        prepared["implementation_backend"] = (
            "custom_single_linkage"
            if clustering_mode == "single_linkage"
            else "sklearn_dbscan"
        )
        return prepared

    def chunk_grouped_units(
        self,
        grouped_units: dict[str, list[dict[str, Any]]],
        config: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        all_chunks: list[dict[str, Any]] = []
        grouped_chunks: list[tuple[str, list[dict[str, Any]]]] = []
        progress = tqdm(
            grouped_units.items(),
            total=len(grouped_units),
            desc=f"Semantic clustering ({config['clustering_mode']}, {config['embedding_model']})",
            unit="doc",
        )

        for doc_id, units in progress:
            ordered_units = self.ordered_units(units)
            chunks = self._build_document_chunks(
                doc_id=doc_id,
                units=ordered_units,
                config=config,
            )
            grouped_chunks.append((doc_id, chunks))
            all_chunks.extend(chunks)

        _, documents = self.build_documents_from_grouped_chunks(
            grouped_units=grouped_units,
            grouped_chunks=grouped_chunks,
            reused=False,
        )
        return all_chunks, documents

    def build_semantic_method_metadata(self, config: dict[str, Any]) -> dict[str, Any]:
        return {
            "paper_method_family": config["paper_method_family"],
            "clustering_mode": config["clustering_mode"],
            "lambda_weight": config["lambda_weight"],
            "stop_distance_threshold": config["stop_distance_threshold"],
            "allow_non_contiguous_chunks": config["allow_non_contiguous_chunks"],
            "number_of_chunks": config["number_of_chunks"],
            "dbscan_eps": config["dbscan_eps"],
            "dbscan_min_samples": config["dbscan_min_samples"],
            "implementation_status": config["implementation_status"],
            "implementation_backend": config["implementation_backend"],
        }

    def semantic_method_metadata_matches(
        self,
        metadata: dict[str, Any],
        config: dict[str, Any],
    ) -> bool:
        return (
            metadata.get("paper_method_family") == config["paper_method_family"]
            and metadata.get("clustering_mode") == config["clustering_mode"]
            and metadata.get("lambda_weight") == config["lambda_weight"]
            and metadata.get("stop_distance_threshold") == config["stop_distance_threshold"]
            and metadata.get("allow_non_contiguous_chunks") == config["allow_non_contiguous_chunks"]
            and metadata.get("number_of_chunks") == config["number_of_chunks"]
            and metadata.get("dbscan_eps") == config["dbscan_eps"]
            and metadata.get("dbscan_min_samples") == config["dbscan_min_samples"]
            and metadata.get("implementation_status") == config["implementation_status"]
            and metadata.get("implementation_backend") == config["implementation_backend"]
        )

    def _build_document_chunks(
        self,
        *,
        doc_id: str,
        units: list[dict[str, Any]],
        config: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if not units:
            return []

        if len(units) == 1:
            return [
                self._build_chunk_from_cluster(
                    doc_id=doc_id,
                    units=units,
                    cluster_indices=[0],
                    position=0,
                    config=config,
                    derived_max_chunk_size=1,
                    cluster_distance=None,
                    cluster_origin="singleton",
                )
            ]

        embeddings = self.get_cached_semantic_embeddings(units, config)
        distance_matrix = joint_position_semantic_distance_matrix(
            embeddings,
            positional_weight=config["lambda_weight"],
        )

        if config["clustering_mode"] == "single_linkage":
            cluster_indices, merge_distances, derived_max_chunk_size = self._run_single_linkage(
                distance_matrix=distance_matrix,
                num_units=len(units),
                number_of_chunks=config["number_of_chunks"],
                stop_distance_threshold=config["stop_distance_threshold"],
            )
            return [
                self._build_chunk_from_cluster(
                    doc_id=doc_id,
                    units=units,
                    cluster_indices=indices,
                    position=position,
                    config=config,
                    derived_max_chunk_size=derived_max_chunk_size,
                    cluster_distance=merge_distances[position],
                    cluster_origin="single_linkage",
                )
                for position, indices in enumerate(cluster_indices)
            ]

        cluster_indices = self._run_dbscan(
            distance_matrix=distance_matrix,
            eps=config["dbscan_eps"],
            min_samples=config["dbscan_min_samples"],
        )
        return [
            self._build_chunk_from_cluster(
                doc_id=doc_id,
                units=units,
                cluster_indices=indices,
                position=position,
                config=config,
                derived_max_chunk_size=None,
                cluster_distance=None,
                cluster_origin="dbscan",
            )
            for position, indices in enumerate(cluster_indices)
        ]

    def _run_single_linkage(
        self,
        *,
        distance_matrix: np.ndarray,
        num_units: int,
        number_of_chunks: int,
        stop_distance_threshold: float,
    ) -> tuple[list[list[int]], list[float | None], int]:
        derived_max_chunk_size = max(1, math.ceil(num_units / number_of_chunks))
        clusters: list[list[int]] = [[idx] for idx in range(num_units)]
        cluster_distances: list[float | None] = [None for _ in range(num_units)]

        while len(clusters) > 1:
            best_pair: tuple[int, int] | None = None
            best_distance: float | None = None

            for left_idx in range(len(clusters) - 1):
                left_cluster = clusters[left_idx]
                for right_idx in range(left_idx + 1, len(clusters)):
                    right_cluster = clusters[right_idx]
                    if len(left_cluster) + len(right_cluster) > derived_max_chunk_size:
                        continue

                    distance = float(
                        np.min(distance_matrix[np.ix_(left_cluster, right_cluster)])
                    )
                    if best_distance is None or distance < best_distance:
                        best_distance = distance
                        best_pair = (left_idx, right_idx)

            if best_pair is None or best_distance is None:
                break

            if best_distance > stop_distance_threshold:
                break

            left_idx, right_idx = best_pair
            merged_cluster = sorted(clusters[left_idx] + clusters[right_idx])
            clusters[left_idx] = merged_cluster
            del clusters[right_idx]
            cluster_distances[left_idx] = best_distance
            del cluster_distances[right_idx]

            paired = sorted(
                zip(clusters, cluster_distances),
                key=lambda item: item[0][0],
            )
            clusters = [cluster for cluster, _ in paired]
            cluster_distances = [distance for _, distance in paired]

        return clusters, cluster_distances, derived_max_chunk_size

    def _run_dbscan(
        self,
        *,
        distance_matrix: np.ndarray,
        eps: float,
        min_samples: int,
    ) -> list[list[int]]:
        clustering = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric="precomputed",
        )
        labels = clustering.fit_predict(distance_matrix)
        grouped: dict[int, list[int]] = {}
        noise_points: list[list[int]] = []

        for sentence_idx, label in enumerate(labels.tolist()):
            if label == -1:
                noise_points.append([sentence_idx])
                continue
            grouped.setdefault(int(label), []).append(sentence_idx)

        ordered_clusters = list(grouped.values()) + noise_points
        ordered_clusters.sort(key=lambda indices: indices[0])
        return ordered_clusters

    def _build_chunk_from_cluster(
        self,
        *,
        doc_id: str,
        units: list[dict[str, Any]],
        cluster_indices: list[int],
        position: int,
        config: dict[str, Any],
        derived_max_chunk_size: int | None,
        cluster_distance: float | None,
        cluster_origin: str,
    ) -> dict[str, Any]:
        span = [units[idx] for idx in sorted(cluster_indices)]
        sentence_indices = [unit["sentence_idx"] for unit in span]
        start_sentence_idx = min(sentence_indices)
        end_sentence_idx = max(sentence_indices)
        sentences = [unit["text"] for unit in span]
        is_contiguous = self._is_contiguous(sentence_indices)

        return {
            "chunk_id": f"{doc_id}::chunk_{position}",
            "doc_id": doc_id,
            "text": self._join_unit_texts(span),
            "sentences": sentences,
            "sentence_indices": sentence_indices,
            "start_sentence_idx": start_sentence_idx,
            "end_sentence_idx": end_sentence_idx,
            "position": position,
            "metadata": {
                "chunking_type": self.chunking_type,
                "embedding_model": config["embedding_model"],
                "similarity_metric": config["similarity_metric"],
                "clustering_mode": config["clustering_mode"],
                "lambda_weight": config["lambda_weight"],
                "number_of_chunks": config["number_of_chunks"],
                "derived_max_chunk_size": derived_max_chunk_size,
                "stop_distance_threshold": config["stop_distance_threshold"],
                "dbscan_eps": config["dbscan_eps"],
                "dbscan_min_samples": config["dbscan_min_samples"],
                "allow_non_contiguous_chunks": config["allow_non_contiguous_chunks"],
                "implementation_backend": config["implementation_backend"],
                "cluster_distance": cluster_distance,
                "cluster_origin": cluster_origin,
                "cluster_size": len(span),
                "is_contiguous": is_contiguous,
                "sentence_indices": sentence_indices,
            },
        }

    @staticmethod
    def _is_contiguous(sentence_indices: list[int]) -> bool:
        if not sentence_indices:
            return True
        return sentence_indices == list(range(sentence_indices[0], sentence_indices[-1] + 1))

    @staticmethod
    def _join_unit_texts(units: list[dict[str, Any]]) -> str:
        return " ".join(unit["text"].strip() for unit in units if unit["text"].strip())
