from __future__ import annotations

from abc import abstractmethod
from typing import Any

import numpy as np

from src.embeddings.base import (
    BaseEmbedder,
    EmbeddingItem,
    build_embedding_output,
    normalize_items,
    validate_embedder_config,
)


class _LangChainSentenceTransformerAdapter:
    def __init__(self, embedder: "SentenceTransformerEmbedder", config: dict[str, Any]) -> None:
        self._embedder = embedder
        self._config = dict(config)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings, _ = self._embedder.encode_texts(texts, self._config)
        return np.asarray(embeddings, dtype=float).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class SentenceTransformerEmbedder(BaseEmbedder):
    MODEL_NAME = ""
    EMBEDDER_NAME = ""
    TRUST_REMOTE_CODE = False
    ALLOW_INSTRUCTION = False

    def __init__(self) -> None:
        self._model = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.MODEL_NAME,
                trust_remote_code=self.TRUST_REMOTE_CODE,
            )
        return self._model

    def _validate_runtime_config(self, config: dict[str, Any]) -> dict[str, Any]:
        return validate_embedder_config(
            config,
            allow_instruction=self.ALLOW_INSTRUCTION,
        )

    def _prepare_texts(
        self,
        texts: list[str],
        validated_config: dict[str, Any],
    ) -> list[str]:
        instruction = validated_config.get("instruction", "")
        if instruction:
            return [f"{instruction}{text}" for text in texts]
        return texts

    def encode_texts(self, texts: list[str], config: dict) -> tuple[Any, dict[str, Any]]:
        validated_config = self._validate_runtime_config(config)
        prepared_texts = self._prepare_texts(texts, validated_config)
        batch_size = validated_config["batch_size"]
        normalize_embeddings = validated_config["normalize_embeddings"]
        show_progress_bar = validated_config["show_progress_bar"]
        convert_to_numpy = validated_config["convert_to_numpy"]
        log_embedding_calls = validated_config["log_embedding_calls"]

        model = self._load_model()

        if log_embedding_calls:
            print(
                f"[EMBEDDING] {self.EMBEDDER_NAME} embedder with model={self.MODEL_NAME}, "
                f"batch_size={batch_size}, items={len(prepared_texts)}"
            )

        embeddings = model.encode(
            prepared_texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=convert_to_numpy,
            normalize_embeddings=normalize_embeddings,
        )

        embedding_dim = int(embeddings.shape[1]) if len(texts) > 0 else 0
        metadata = {
            "embedder": self.EMBEDDER_NAME,
            "model_name": self.MODEL_NAME,
            "num_items": len(texts),
            "batch_size": batch_size,
            "normalize_embeddings": normalize_embeddings,
            "show_progress_bar": show_progress_bar,
            "convert_to_numpy": convert_to_numpy,
            "embedding_dim": embedding_dim,
        }
        if self.ALLOW_INSTRUCTION:
            metadata["instruction"] = validated_config["instruction"]
        return embeddings, metadata

    def encode_items(
        self,
        items: list[EmbeddingItem],
        config: dict,
    ) -> tuple[Any, list[EmbeddingItem], dict[str, Any]]:
        embeddings, metadata = self.encode_texts([item.text for item in items], config)
        return embeddings, items, metadata

    def as_langchain_embeddings(self, config: dict[str, Any]) -> Any:
        return _LangChainSentenceTransformerAdapter(self, config)

    def embed(self, data: Any, config: dict) -> dict[str, Any]:
        items = normalize_items(data)
        embeddings, items, metadata = self.encode_items(items, config)
        return build_embedding_output(
            embeddings=embeddings,
            items=items,
            metadata=metadata,
        )

    @classmethod
    @abstractmethod
    def canonical_model_name(cls) -> str:
        raise NotImplementedError
