from __future__ import annotations

from src.embeddings.sentence_transformer_base import SentenceTransformerEmbedder


class BGEEmbedder(SentenceTransformerEmbedder):
    MODEL_NAME = "BAAI/bge-large-en-v1.5"
    EMBEDDER_NAME = "bge"
    ALLOW_INSTRUCTION = True

    @classmethod
    def canonical_model_name(cls) -> str:
        return cls.MODEL_NAME
