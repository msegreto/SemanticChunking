from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict

import numpy as np

from src.retrieval.base import BaseRetriever


class NumpyRetriever(BaseRetriever):
    def __init__(self) -> None:
        self.vectors = None
        self.metadata = None
        self.distance = "cosine"
        self.normalize = True

    def build_index(self, embedding_output: Any, config: Dict, global_config: Dict) -> Dict[str, Any]:
        if not isinstance(embedding_output, dict):
            raise TypeError("embedding_output must be a dict.")

        embeddings = embedding_output.get("embeddings")
        items = embedding_output.get("items", [])
        emb_metadata = embedding_output.get("metadata", {})

        if embeddings is None:
            raise ValueError("embedding_output does not contain 'embeddings'.")

        vectors = np.asarray(embeddings, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError(f"Embeddings must be a 2D array. Got shape: {vectors.shape}")

        self.distance = config.get("distance", "cosine").lower()
        self.normalize = config.get("normalize", True)

        dataset_name = global_config["dataset"]["name"]
        chunking_type = global_config["chunking"]["type"]
        embedder_name = global_config["embedding"]["name"]

        output_dir = config.get(
            "output_dir",
            f"data/indexes/{dataset_name}/{chunking_type}/{embedder_name}",
        )
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if self.normalize or self.distance == "cosine":
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms = np.clip(norms, 1e-12, None)
            vectors = vectors / norms

        print(
            f"[RETRIEVAL] Building NumPy index with distance={self.distance}, "
            f"normalize={self.normalize}, vectors={vectors.shape[0]}, dim={vectors.shape[1]}"
        )

        vectors_path = output_path / "vectors.npy"
        metadata_path = output_path / "metadata.pkl"
        manifest_path = output_path / "manifest.json"

        np.save(vectors_path, vectors)

        serializable_items = []
        for item in items:
            if hasattr(item, "__dict__"):
                serializable_items.append(item.__dict__)
            else:
                serializable_items.append(item)

        with metadata_path.open("wb") as f:
            pickle.dump(
                {
                    "items": serializable_items,
                    "embedding_metadata": emb_metadata,
                    "distance": self.distance,
                    "normalize": self.normalize,
                    "num_vectors": int(vectors.shape[0]),
                    "dimension": int(vectors.shape[1]),
                },
                f,
            )

        manifest = {
            "retriever": "numpy",
            "dataset": dataset_name,
            "chunking": chunking_type,
            "embedder": embedder_name,
            "vectors_path": str(vectors_path),
            "metadata_path": str(metadata_path),
            "distance": self.distance,
            "normalize": self.normalize,
            "num_vectors": int(vectors.shape[0]),
            "dimension": int(vectors.shape[1]),
        }

        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        self.vectors = vectors
        self.metadata = serializable_items

        return {
            "index_path": str(vectors_path),  # lasciato così per compatibilità con orchestrator
            "metadata_path": str(metadata_path),
            "manifest_path": str(manifest_path),
            "num_vectors": int(vectors.shape[0]),
            "dimension": int(vectors.shape[1]),
            "distance": self.distance,
        }

    def load_index(self, index_path: str, metadata_path: str | None = None) -> Dict[str, Any]:
        vectors = np.load(index_path)
        metadata = None

        if metadata_path is not None:
            with open(metadata_path, "rb") as f:
                metadata = pickle.load(f)

        self.vectors = vectors
        self.metadata = metadata

        return {
            "index": vectors,
            "metadata": metadata,
        }

    def search(self, query_vectors: Any, top_k: int = 10) -> Dict[str, Any]:
        if self.vectors is None:
            raise RuntimeError("Index not loaded. Call load_index() or build_index() first.")

        queries = np.asarray(query_vectors, dtype=np.float32)
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)

        if self.normalize or self.distance == "cosine":
            norms = np.linalg.norm(queries, axis=1, keepdims=True)
            norms = np.clip(norms, 1e-12, None)
            queries = queries / norms

        if self.distance == "cosine":
            scores = queries @ self.vectors.T
            top_idx = np.argsort(-scores, axis=1)[:, :top_k]
            top_scores = np.take_along_axis(scores, top_idx, axis=1)

        elif self.distance == "l2":
            # dists: (n_queries, n_docs)
            dists = np.sum((queries[:, None, :] - self.vectors[None, :, :]) ** 2, axis=2)
            top_idx = np.argsort(dists, axis=1)[:, :top_k]
            top_scores = np.take_along_axis(dists, top_idx, axis=1)

        else:
            raise ValueError("Unsupported distance. Use 'cosine' or 'l2'.")

        return {
            "scores": top_scores,
            "indices": top_idx,
            "metadata": self.metadata,
        }