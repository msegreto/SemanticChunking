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
        self._stream_state: dict[str, Any] | None = None

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

    def start_incremental_index(self, config: Dict, global_config: Dict) -> None:
        resume = bool(config.get("_stream_resume", False))
        dataset_name = global_config["dataset"]["name"]
        chunking_type = global_config["chunking"]["type"]
        embedder_name = global_config["embedding"]["name"]
        normalize = bool(config.get("normalize", True))
        distance = config.get("distance", "cosine").lower()

        output_dir = config.get(
            "output_dir",
            f"data/indexes/{dataset_name}/{chunking_type}/{embedder_name}",
        )
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        items_jsonl_path = output_path / "items.jsonl"
        if items_jsonl_path.exists() and not resume:
            items_jsonl_path.unlink()

        index_path = output_path / "index.faiss"
        metadata_path = output_path / "metadata.pkl"
        existing_vectors = 0
        existing_items = 0
        existing_dim = None
        existing_embedding_metadata = None
        existing_index = None
        if resume and index_path.exists() and metadata_path.exists():
            try:
                import faiss
                existing_index = faiss.read_index(str(index_path))
            except Exception:
                existing_index = None
            try:
                previous = pickle.loads(metadata_path.read_bytes())
            except Exception:
                previous = None
            if isinstance(previous, dict):
                existing_vectors = int(previous.get("num_vectors", 0) or 0)
                existing_items = int(previous.get("num_items", 0) or 0)
                dim = previous.get("dimension")
                if isinstance(dim, int) and dim >= 0:
                    existing_dim = dim
                emb_meta = previous.get("embedding_metadata")
                if isinstance(emb_meta, dict):
                    existing_embedding_metadata = dict(emb_meta)

        self._stream_state = {
            "output_path": output_path,
            "items_jsonl_path": items_jsonl_path,
            "dataset_name": dataset_name,
            "chunking_type": chunking_type,
            "embedder_name": embedder_name,
            "normalize": normalize,
            "distance": distance,
            "dimension": existing_dim,
            "num_vectors": 0,
            "num_items": 0,
            "embedding_metadata": existing_embedding_metadata,
            "existing_vectors": existing_vectors,
            "existing_items": existing_items,
            "index_path": index_path,
            "metadata_path": metadata_path,
        }
        self.index = existing_index

    def add_embeddings_batch(
        self,
        vectors: Any,
        items: list[dict[str, Any]],
        embedding_metadata: Dict[str, Any] | None = None,
    ) -> None:
        if self._stream_state is None:
            raise RuntimeError("start_incremental_index() must be called before add_embeddings_batch().")

        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"Batch embeddings must be 2D. Got shape {arr.shape}.")
        if arr.shape[0] != len(items):
            raise ValueError(
                f"Batch embeddings/items mismatch: {arr.shape[0]} vectors vs {len(items)} items."
            )
        if arr.shape[0] == 0:
            return

        try:
            import faiss
        except ImportError as e:
            raise ImportError(
                "faiss is not installed. Install faiss-cpu or faiss-gpu before running retrieval indexing."
            ) from e

        if self._stream_state["normalize"] or self._stream_state["distance"] == "cosine":
            faiss.normalize_L2(arr)

        dim = int(arr.shape[1])
        expected_dim = self._stream_state["dimension"]
        if expected_dim is None:
            self._stream_state["dimension"] = dim
            if self._stream_state["distance"] == "cosine":
                self.index = faiss.IndexFlatIP(dim)
            elif self._stream_state["distance"] == "l2":
                self.index = faiss.IndexFlatL2(dim)
            else:
                raise ValueError("Unsupported distance. Use 'cosine' or 'l2'.")
        elif int(expected_dim) != dim:
            raise ValueError(
                f"Inconsistent embedding dimension across batches: expected {expected_dim}, got {dim}"
            )

        self.index.add(arr)
        self._stream_state["num_vectors"] += int(arr.shape[0])
        self._stream_state["num_items"] += len(items)
        if embedding_metadata and self._stream_state["embedding_metadata"] is None:
            self._stream_state["embedding_metadata"] = dict(embedding_metadata)

        with self._stream_state["items_jsonl_path"].open("a", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def finalize_incremental_index(self) -> Dict[str, Any]:
        if self._stream_state is None:
            raise RuntimeError("No incremental indexing session found.")

        try:
            import faiss
        except ImportError as e:
            raise ImportError(
                "faiss is not installed. Install faiss-cpu or faiss-gpu before running retrieval indexing."
            ) from e

        output_path: Path = self._stream_state["output_path"]
        index_path: Path = self._stream_state["index_path"]
        metadata_path: Path = self._stream_state["metadata_path"]
        manifest_path = output_path / "manifest.json"
        items_jsonl_path: Path = self._stream_state["items_jsonl_path"]

        num_vectors = int(self._stream_state["num_vectors"]) + int(self._stream_state.get("existing_vectors", 0))
        num_items = int(self._stream_state["num_items"]) + int(self._stream_state.get("existing_items", 0))
        dim = int(self._stream_state["dimension"] or 0)
        emb_metadata = self._stream_state["embedding_metadata"] or {}

        if self.index is None:
            if self._stream_state["distance"] == "cosine":
                self.index = faiss.IndexFlatIP(max(dim, 1))
            else:
                self.index = faiss.IndexFlatL2(max(dim, 1))
        faiss.write_index(self.index, str(index_path))

        with metadata_path.open("wb") as f:
            pickle.dump(
                {
                    "items_jsonl_path": str(items_jsonl_path),
                    "num_items": num_items,
                    "embedding_metadata": emb_metadata,
                    "distance": self._stream_state["distance"],
                    "normalize": self._stream_state["normalize"],
                    "num_vectors": num_vectors,
                    "dimension": dim,
                },
                f,
            )

        manifest = {
            "schema_version": 1,
            "retriever": "faiss",
            "dataset": self._stream_state["dataset_name"],
            "chunking": self._stream_state["chunking_type"],
            "embedder": self._stream_state["embedder_name"],
            "model_name": emb_metadata.get("model_name"),
            "index_path": str(index_path),
            "metadata_path": str(metadata_path),
            "manifest_path": str(manifest_path),
            "distance": self._stream_state["distance"],
            "normalize": self._stream_state["normalize"],
            "num_vectors": num_vectors,
            "dimension": dim,
            "num_items": num_items,
        }
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        self.metadata = {
            "items_jsonl_path": str(items_jsonl_path),
            "num_items": num_items,
        }
        self._stream_state = None

        return {
            "index_path": str(index_path),
            "metadata_path": str(metadata_path),
            "manifest_path": str(manifest_path),
            "num_vectors": num_vectors,
            "dimension": dim,
            "distance": manifest["distance"],
        }
