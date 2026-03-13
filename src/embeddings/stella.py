from __future__ import annotations

from src.embeddings.sentence_transformer_base import SentenceTransformerEmbedder


class StellaEmbedder(SentenceTransformerEmbedder):
    MODEL_NAME = "dunzhang/stella_en_1.5B_v5"
    EMBEDDER_NAME = "stella"
    TRUST_REMOTE_CODE = True

    @classmethod
    def canonical_model_name(cls) -> str:
        return cls.MODEL_NAME
