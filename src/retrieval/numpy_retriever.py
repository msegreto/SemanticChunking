from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict

import numpy as np

from src.embeddings.base import validate_embedding_output
from src.retrieval.base import BaseRetriever


class NumpyRetriever(BaseRetriever):
    def __init__(self) -> None:
        self.vectors = None
        self.metadata = None
        self.distance = "cosine"
        self.normalize = True
        self._stream_state: dict[str, Any] | None = None

    def build_index(self, embedding_output: Any, config: Dict, global_config: Dict) -> Dict[str, Any]:
        validated_output = validate_embedding_output(embedding_output)
        embeddings = validated_output["embeddings"]
        items = validated_output["items"]
        emb_metadata = validated_output["metadata"]

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
            "schema_version": 1,
            "retriever": "numpy",
            "dataset": dataset_name,
            "chunking": chunking_type,
            "embedder": embedder_name,
            "model_name": emb_metadata.get("model_name"),
            "index_path": str(vectors_path),
            "vectors_path": str(vectors_path),
            "metadata_path": str(metadata_path),
            "manifest_path": str(manifest_path),
            "distance": self.distance,
            "normalize": self.normalize,
            "num_vectors": int(vectors.shape[0]),
            "dimension": int(vectors.shape[1]),
            "num_items": len(items),
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
        vectors = np.load(index_path, mmap_mode="r")
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

    def start_incremental_index(self, config: Dict, global_config: Dict) -> None:
        self.distance = config.get("distance", "cosine").lower()
        self.normalize = bool(config.get("normalize", True))
        resume = bool(config.get("_stream_resume", False))

        dataset_name = global_config["dataset"]["name"]
        chunking_type = global_config["chunking"]["type"]
        embedder_name = global_config["embedding"]["name"]

        output_dir = config.get(
            "output_dir",
            f"data/indexes/{dataset_name}/{chunking_type}/{embedder_name}",
        )
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        temp_dir = output_path / "_stream_batches"
        temp_dir.mkdir(parents=True, exist_ok=True)
        items_jsonl_path = output_path / "items.jsonl"
        if items_jsonl_path.exists() and not resume:
            items_jsonl_path.unlink()

        vectors_path = output_path / "vectors.npy"
        metadata_path = output_path / "metadata.pkl"

        existing_vectors = 0
        existing_items = 0
        existing_dim = None
        existing_embedding_metadata = None

        if resume and vectors_path.exists() and metadata_path.exists():
            previous = pickle.loads(metadata_path.read_bytes())
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
            "temp_dir": temp_dir,
            "items_jsonl_path": items_jsonl_path,
            "batch_paths": [],
            "num_vectors": 0,
            "num_items": 0,
            "dimension": existing_dim,
            "embedding_metadata": existing_embedding_metadata,
            "dataset_name": dataset_name,
            "chunking_type": chunking_type,
            "embedder_name": embedder_name,
            "vectors_path": vectors_path,
            "metadata_path": metadata_path,
            "resume": resume,
            "existing_vectors": existing_vectors,
            "existing_items": existing_items,
        }

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

        if self.normalize or self.distance == "cosine":
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms = np.clip(norms, 1e-12, None)
            arr = arr / norms

        expected_dim = self._stream_state["dimension"]
        if expected_dim is None:
            self._stream_state["dimension"] = int(arr.shape[1])
        elif int(arr.shape[1]) != int(expected_dim):
            raise ValueError(
                f"Inconsistent embedding dimension across batches: "
                f"expected {expected_dim}, got {arr.shape[1]}"
            )

        batch_id = len(self._stream_state["batch_paths"])
        batch_path = self._stream_state["temp_dir"] / f"batch_{batch_id:08d}.npy"
        np.save(batch_path, arr)
        self._stream_state["batch_paths"].append(batch_path)

        with self._stream_state["items_jsonl_path"].open("a", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        self._stream_state["num_vectors"] += int(arr.shape[0])
        self._stream_state["num_items"] += len(items)
        if embedding_metadata and self._stream_state["embedding_metadata"] is None:
            self._stream_state["embedding_metadata"] = dict(embedding_metadata)

    def finalize_incremental_index(self) -> Dict[str, Any]:
        if self._stream_state is None:
            raise RuntimeError("No incremental indexing session found.")

        new_vectors = int(self._stream_state["num_vectors"])
        new_items = int(self._stream_state["num_items"])
        existing_vectors = int(self._stream_state.get("existing_vectors", 0))
        existing_items = int(self._stream_state.get("existing_items", 0))
        num_vectors = existing_vectors + new_vectors
        num_items = existing_items + new_items
        dim = self._stream_state["dimension"]
        if dim is None:
            dim = 0

        output_path: Path = self._stream_state["output_path"]
        vectors_path: Path = self._stream_state["vectors_path"]
        metadata_path: Path = self._stream_state["metadata_path"]
        manifest_path = output_path / "manifest.json"
        items_jsonl_path: Path = self._stream_state["items_jsonl_path"]

        if num_vectors > 0:
            from numpy.lib.format import open_memmap

            tmp_vectors_path = vectors_path.with_name(f"{vectors_path.stem}.next.npy")
            dst = open_memmap(
                tmp_vectors_path,
                mode="w+",
                dtype=np.float32,
                shape=(num_vectors, int(dim)),
            )
            cursor = 0
            if existing_vectors > 0 and vectors_path.exists():
                old = np.load(vectors_path, mmap_mode="r")
                dst[:existing_vectors] = old
                cursor = existing_vectors
            for batch_path in self._stream_state["batch_paths"]:
                batch = np.load(batch_path, mmap_mode="r")
                rows = int(batch.shape[0])
                dst[cursor:cursor + rows] = batch
                cursor += rows
            del dst
            tmp_vectors_path.replace(vectors_path)
        else:
            np.save(vectors_path, np.zeros((0, 0), dtype=np.float32))

        emb_metadata = self._stream_state["embedding_metadata"] or {}
        with metadata_path.open("wb") as f:
            pickle.dump(
                {
                    "items_jsonl_path": str(items_jsonl_path),
                    "num_items": num_items,
                    "embedding_metadata": emb_metadata,
                    "distance": self.distance,
                    "normalize": self.normalize,
                    "num_vectors": num_vectors,
                    "dimension": int(dim),
                },
                f,
            )

        manifest = {
            "schema_version": 1,
            "retriever": "numpy",
            "dataset": self._stream_state["dataset_name"],
            "chunking": self._stream_state["chunking_type"],
            "embedder": self._stream_state["embedder_name"],
            "model_name": emb_metadata.get("model_name"),
            "index_path": str(vectors_path),
            "vectors_path": str(vectors_path),
            "metadata_path": str(metadata_path),
            "manifest_path": str(manifest_path),
            "distance": self.distance,
            "normalize": self.normalize,
            "num_vectors": num_vectors,
            "dimension": int(dim),
            "num_items": num_items,
        }
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        for batch_path in self._stream_state["batch_paths"]:
            try:
                batch_path.unlink()
            except OSError:
                pass
        try:
            self._stream_state["temp_dir"].rmdir()
        except OSError:
            pass

        self.vectors = np.load(vectors_path, mmap_mode="r")
        self.metadata = {
            "items_jsonl_path": str(items_jsonl_path),
            "num_items": num_items,
        }
        self._stream_state = None

        return {
            "index_path": str(vectors_path),
            "metadata_path": str(metadata_path),
            "manifest_path": str(manifest_path),
            "num_vectors": num_vectors,
            "dimension": int(dim),
            "distance": self.distance,
        }
