from __future__ import annotations

from typing import Any, List

from src.embeddings.base import BaseEmbedder, EmbeddingResult, normalize_items


class BGEEmbedder(BaseEmbedder):
    MODEL_NAME = "BAAI/bge-large-en-v1.5"

    def __init__(self) -> None:
        self._model = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.MODEL_NAME)
        return self._model

    def embed(self, data: Any, config: dict) -> EmbeddingResult:
        items = normalize_items(data)
        texts: List[str] = [item.text for item in items]

        batch_size = config.get("batch_size", 16)
        normalize_embeddings = config.get("normalize_embeddings", True)
        show_progress_bar = config.get("show_progress_bar", False)
        convert_to_numpy = config.get("convert_to_numpy", True)

        # Opzionale: utile soprattutto per query embedding, ma lo lascio supportato
        instruction = config.get("instruction", "")
        if instruction:
            texts = [f"{instruction}{text}" for text in texts]

        model = self._load_model()

        print(
            f"[EMBEDDING] BGE embedder with model={self.MODEL_NAME}, "
            f"batch_size={batch_size}, items={len(texts)}"
        )

        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=convert_to_numpy,
            normalize_embeddings=normalize_embeddings,
        )

        return {
            "embeddings": embeddings,
            "items": items,
            "metadata": {
                "embedder": "bge",
                "model_name": self.MODEL_NAME,
                "num_items": len(items),
                "batch_size": batch_size,
                "normalize_embeddings": normalize_embeddings,
            },
        }