from __future__ import annotations

import json
from abc import abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from src.chunking.base import BaseChunker
from src.chunking.semantic_utils import (
    build_common_semantic_metadata,
    sort_units_by_sentence_idx,
    validate_common_semantic_config,
)
from src.embeddings.base import BaseEmbedder
from src.embeddings.factory import EmbedderFactory


class BaseSemanticChunker(BaseChunker):
    def __init__(self) -> None:
        self._semantic_embedder: BaseEmbedder | None = None
        self._semantic_embedder_name: str | None = None

    def validate_strategy_config(self, config: dict[str, Any]) -> dict[str, Any]:
        validated = validate_common_semantic_config(config)
        return self.validate_semantic_method_config(validated)

    def prepare_chunking_inputs(self, routed_output: Any, config: dict[str, Any]) -> dict[str, Any]:
        prepared = super().prepare_chunking_inputs(routed_output, config)
        split_metadata = routed_output.get("metadata", {}) if isinstance(routed_output, dict) else {}
        self.ensure_semantic_embeddings(
            split_units=prepared["split_units"],
            grouped_units=prepared["grouped_units"],
            config=prepared["config"],
            split_path=split_metadata.get("split_path"),
        )
        return prepared

    def saved_payload_matches(
        self,
        metadata: Any,
        doc_id: str,
        config: dict[str, Any],
    ) -> bool:
        if not isinstance(metadata, dict):
            return False

        return (
            metadata.get("chunking_type") == self.chunking_type
            and metadata.get("dataset") == config["dataset"]
            and metadata.get("doc_id") == doc_id
            and metadata.get("yaml_name") == config["yaml_name"]
            and metadata.get("embedding_model") == config["embedding_model"]
            and metadata.get("similarity_metric") == config["similarity_metric"]
            and metadata.get("embedding_batch_size") == config["embedding_batch_size"]
            and self.semantic_method_metadata_matches(metadata, config)
        )

    def build_saved_payload_metadata(
        self,
        doc_id: str,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "chunking_type": self.chunking_type,
            "dataset": config["dataset"],
            "doc_id": doc_id,
            "yaml_name": config["yaml_name"],
            **build_common_semantic_metadata(config),
            **self.build_semantic_method_metadata(config),
        }

    def build_strategy_config_metadata(self, config: dict[str, Any]) -> dict[str, Any]:
        return {
            **build_common_semantic_metadata(config),
            **self.build_semantic_method_metadata(config),
        }

    def ordered_units(self, units: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sort_units_by_sentence_idx(units)

    def get_semantic_embedder(self, config: dict[str, Any]) -> BaseEmbedder:
        requested_name = config["embedding_model"]
        if self._semantic_embedder is None or self._semantic_embedder_name != requested_name:
            self._semantic_embedder = EmbedderFactory.create(requested_name)
            self._semantic_embedder_name = requested_name
        return self._semantic_embedder

    def build_semantic_embedder_config(self, config: dict[str, Any]) -> dict[str, Any]:
        return {
            "batch_size": config["embedding_batch_size"],
            "normalize_embeddings": True,
            "show_progress_bar": config["show_embedding_progress"],
            "convert_to_numpy": True,
            "log_embedding_calls": False,
        }

    def encode_semantic_texts(
        self,
        texts: list[str],
        config: dict[str, Any],
    ) -> tuple[Any, dict[str, Any]]:
        embedder = self.get_semantic_embedder(config)
        return embedder.encode_texts(
            texts,
            self.build_semantic_embedder_config(config),
        )

    def build_semantic_embedding_texts(
        self,
        units: list[dict[str, Any]],
        config: dict[str, Any],
    ) -> list[str]:
        return [unit["text"].strip() for unit in units]

    def semantic_embedding_cache_key(self, config: dict[str, Any]) -> str:
        return config["embedding_model"]

    def ensure_semantic_embeddings(
        self,
        *,
        split_units: list[dict[str, Any]],
        grouped_units: dict[str, list[dict[str, Any]]],
        config: dict[str, Any],
        split_path: str | None,
    ) -> None:
        cache_key = self.semantic_embedding_cache_key(config)
        expected_texts_by_unit = self._build_expected_texts_by_unit(grouped_units, config)

        if self._all_units_have_cached_embeddings(split_units, expected_texts_by_unit, cache_key):
            print(f"[CHUNKING] reusing sentence embeddings from split cache for {cache_key}")
            return

        texts = [expected_texts_by_unit[id(unit)] for unit in split_units]
        embeddings, _ = self.encode_semantic_texts(texts, config)
        matrix = np.asarray(embeddings, dtype=float)

        for unit, vector in zip(split_units, matrix):
            unit.setdefault("embeddings", {})
            unit["embeddings"][cache_key] = {
                "text": expected_texts_by_unit[id(unit)],
                "vector": vector.tolist(),
            }

        if split_path:
            self._save_split_units(Path(split_path), split_units)

    def get_cached_semantic_embeddings(
        self,
        units: list[dict[str, Any]],
        config: dict[str, Any],
    ) -> np.ndarray:
        ordered_units = self.ordered_units(units)
        expected_texts = self.build_semantic_embedding_texts(ordered_units, config)
        cache_key = self.semantic_embedding_cache_key(config)
        vectors: list[list[float]] = []

        for unit, expected_text in zip(ordered_units, expected_texts):
            cache_entries = unit.get("embeddings", {})
            if not isinstance(cache_entries, dict):
                raise ValueError(f"Missing cached semantic embeddings for model '{cache_key}'.")

            cache_entry = cache_entries.get(cache_key)
            if not isinstance(cache_entry, dict):
                raise ValueError(f"Missing cached semantic embedding for model '{cache_key}'.")
            if cache_entry.get("text") != expected_text:
                raise ValueError(
                    f"Cached semantic embedding text mismatch for model '{cache_key}'."
                )

            vector = cache_entry.get("vector")
            if not isinstance(vector, list):
                raise ValueError(
                    f"Cached semantic embedding vector is invalid for model '{cache_key}'."
                )
            vectors.append(vector)

        return np.asarray(vectors, dtype=float)

    def _build_expected_texts_by_unit(
        self,
        grouped_units: dict[str, list[dict[str, Any]]],
        config: dict[str, Any],
    ) -> dict[int, str]:
        expected_texts_by_unit: dict[int, str] = {}
        for units in grouped_units.values():
            ordered_units = self.ordered_units(units)
            embedding_texts = self.build_semantic_embedding_texts(ordered_units, config)
            if len(embedding_texts) != len(ordered_units):
                raise ValueError("Semantic embedding texts count must match the number of units.")
            for unit, text in zip(ordered_units, embedding_texts):
                expected_texts_by_unit[id(unit)] = text
        return expected_texts_by_unit

    def _all_units_have_cached_embeddings(
        self,
        split_units: list[dict[str, Any]],
        expected_texts_by_unit: dict[int, str],
        cache_key: str,
    ) -> bool:
        for unit in split_units:
            cache_entries = unit.get("embeddings")
            if not isinstance(cache_entries, dict):
                return False

            cache_entry = cache_entries.get(cache_key)
            if not isinstance(cache_entry, dict):
                return False
            if cache_entry.get("text") != expected_texts_by_unit[id(unit)]:
                return False

            vector = cache_entry.get("vector")
            if not isinstance(vector, list) or not vector:
                return False

        return True

    def _save_split_units(self, split_path: Path, split_units: list[dict[str, Any]]) -> None:
        split_path.parent.mkdir(parents=True, exist_ok=True)
        with split_path.open("w", encoding="utf-8") as file_obj:
            for item in split_units:
                file_obj.write(json.dumps(item, ensure_ascii=False) + "\n")

    @abstractmethod
    def validate_semantic_method_config(self, config: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def build_semantic_method_metadata(self, config: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def semantic_method_metadata_matches(
        self,
        metadata: dict[str, Any],
        config: dict[str, Any],
    ) -> bool:
        raise NotImplementedError
