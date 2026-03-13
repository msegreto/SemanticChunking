from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from src.embeddings.base import validate_embedding_output
from src.retrieval.base import BaseRetriever


class FAISSRetriever(BaseRetriever):
    def __init__(self) -> None:
        self.index = None
        self.metadata = None

    def build_index(self, embedding_output: Any, config: Dict, global_config: Dict) -> Dict[str, Any]:
        try:
            import faiss
        except ImportError as e:
            raise ImportError(
                "faiss is not installed. Install faiss-cpu or faiss-gpu before running retrieval indexing."
            ) from e

        validated_output = validate_embedding_output(embedding_output)
        embeddings = validated_output["embeddings"]
        items = validated_output["items"]
        emb_metadata = validated_output["metadata"]

        vectors = np.asarray(embeddings, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError(f"Embeddings must be a 2D array. Got shape: {vectors.shape}")

        normalize = config.get("normalize", True)
        distance = config.get("distance", "cosine").lower()
        dataset_name = global_config["dataset"]["name"]
        chunking_type = global_config["chunking"]["type"]
        embedder_name = global_config["embedding"]["name"]

        output_dir = config.get(
            "output_dir",
            f"data/indexes/{dataset_name}/{chunking_type}/{embedder_name}",
        )
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if normalize or distance == "cosine":
            faiss.normalize_L2(vectors)

        dim = vectors.shape[1]

        if distance == "cosine":
            index = faiss.IndexFlatIP(dim)
        elif distance == "l2":
            index = faiss.IndexFlatL2(dim)
        else:
            raise ValueError("Unsupported distance. Use 'cosine' or 'l2'.")

        print(
            f"[RETRIEVAL] Building FAISS index with distance={distance}, "
            f"normalize={normalize}, vectors={vectors.shape[0]}, dim={vectors.shape[1]}"
        )

        index.add(vectors)

        index_path = output_path / "index.faiss"
        metadata_path = output_path / "metadata.pkl"
        manifest_path = output_path / "manifest.json"

        faiss.write_index(index, str(index_path))

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
                    "distance": distance,
                    "normalize": normalize,
                    "num_vectors": int(vectors.shape[0]),
                    "dimension": int(vectors.shape[1]),
                },
                f,
            )

        manifest = {
            "schema_version": 1,
            "retriever": "faiss",
            "dataset": dataset_name,
            "chunking": chunking_type,
            "embedder": embedder_name,
            "model_name": emb_metadata.get("model_name"),
            "index_path": str(index_path),
            "metadata_path": str(metadata_path),
            "manifest_path": str(manifest_path),
            "distance": distance,
            "normalize": normalize,
            "num_vectors": int(vectors.shape[0]),
            "dimension": int(vectors.shape[1]),
            "num_items": len(items),
        }

        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        self.index = index
        self.metadata = serializable_items

        return {
            "index_path": str(index_path),
            "metadata_path": str(metadata_path),
            "manifest_path": str(manifest_path),
            "num_vectors": int(vectors.shape[0]),
            "dimension": int(vectors.shape[1]),
            "distance": distance,
        }

    def load_index(self, index_path: str, metadata_path: str | None = None) -> Dict[str, Any]:
        try:
            import faiss
        except ImportError as e:
            raise ImportError(
                "faiss is not installed. Install faiss-cpu or faiss-gpu before running retrieval indexing."
            ) from e

        index = faiss.read_index(index_path)
        metadata = None

        if metadata_path is not None:
            with open(metadata_path, "rb") as f:
                metadata = pickle.load(f)

        self.index = index
        self.metadata = metadata

        return {
            "index": index,
            "metadata": metadata,
        }

    def search(self, query_vectors: Any, top_k: int = 10) -> Dict[str, Any]:
        if self.index is None:
            raise RuntimeError("Index not loaded. Call load_index() or build_index() first.")

        vectors = np.asarray(query_vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        try:
            import faiss
            faiss.normalize_L2(vectors)
        except Exception:
            pass

        scores, indices = self.index.search(vectors, top_k)

        return {
            "scores": scores,
            "indices": indices,
            "metadata": self.metadata,
        }
