from __future__ import annotations

from src.embeddings.sentence_transformer_base import SentenceTransformerEmbedder


class MPNetEmbedder(SentenceTransformerEmbedder):
    #MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L3-v2"
    MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
    EMBEDDER_NAME = "mpnet"

    @classmethod
    def canonical_model_name(cls) -> str:
        return cls.MODEL_NAME
