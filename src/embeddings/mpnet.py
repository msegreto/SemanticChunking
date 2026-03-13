from __future__ import annotations

from typing import Any, List

from src.embeddings.base import (
    BaseEmbedder,
    build_embedding_output,
    normalize_items,
    validate_embedder_config,
)


class MPNetEmbedder(BaseEmbedder):
    MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
    # MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L3-v2"

    def __init__(self) -> None:
        self._model = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.MODEL_NAME)
        return self._model

    def embed(self, data: Any, config: dict) -> dict[str, Any]:
        items = normalize_items(data)
        texts: List[str] = [item.text for item in items]

        validated_config = validate_embedder_config(config)
        batch_size = validated_config["batch_size"]
        normalize_embeddings = validated_config["normalize_embeddings"]
        show_progress_bar = validated_config["show_progress_bar"]
        convert_to_numpy = validated_config["convert_to_numpy"]

        model = self._load_model()

        print(
            f"[EMBEDDING] MPNet embedder with model={self.MODEL_NAME}, "
            f"batch_size={batch_size}, items={len(texts)}"
        )

        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=convert_to_numpy,
            normalize_embeddings=normalize_embeddings,
        )

        embedding_dim = int(embeddings.shape[1]) if len(items) > 0 else 0

        return build_embedding_output(
            embeddings=embeddings,
            items=items,
            metadata={
                "embedder": "mpnet",
                "model_name": self.MODEL_NAME,
                "num_items": len(items),
                "batch_size": batch_size,
                "normalize_embeddings": normalize_embeddings,
                "show_progress_bar": show_progress_bar,
                "convert_to_numpy": convert_to_numpy,
                "embedding_dim": embedding_dim,
            },
        )
