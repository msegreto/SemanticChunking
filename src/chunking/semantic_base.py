from __future__ import annotations

from abc import abstractmethod
from typing import Any

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
            "show_progress_bar": False,
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
