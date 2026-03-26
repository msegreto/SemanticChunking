from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import pyterrier as pt
except ImportError:  # pragma: no cover
    pt = None

from src.embeddings.factory import EmbedderFactory


_PtTransformerBase = pt.Transformer if pt is not None else object


class RetrievalTransformer(_PtTransformerBase):
    def __init__(
        self,
        *,
        retrieval_config: dict[str, Any],
        dataset_name: str,
        chunking_name: str,
        run_name: str,
        allow_reuse: bool = True,
        force_rebuild: bool = False,
    ) -> None:
        self.retrieval_config = dict(retrieval_config)
        self.dataset_name = str(dataset_name).strip()
        self.chunking_name = str(chunking_name).strip()
        self.run_name = str(run_name).strip() or "default_run"
        self.allow_reuse = bool(allow_reuse)
        self.force_rebuild = bool(force_rebuild)

    @classmethod
    def from_config(
        cls,
        *,
        retrieval_config: dict[str, Any],
        dataset_name: str,
        chunking_name: str,
        run_name: str,
        allow_reuse: bool = True,
        force_rebuild: bool = False,
    ) -> "RetrievalTransformer":
        return cls(
            retrieval_config=retrieval_config,
            dataset_name=dataset_name,
            chunking_name=chunking_name,
            run_name=run_name,
            allow_reuse=allow_reuse,
            force_rebuild=force_rebuild,
        )

    def transform(self, inp: Any) -> dict[str, Any]:
        return self.build(inp)

    def build(self, inp: Any) -> dict[str, Any]:
        chunk_output, dataset = self._resolve_inputs(inp)
        if not self.force_rebuild and self.allow_reuse:
            reusable_output = self.try_load_reusable_output(
                chunk_output=chunk_output,
                dataset=dataset,
            )
            if reusable_output is not None:
                return reusable_output
        index_dir = self.resolve_index_dir()
        index_dir.mkdir(parents=True, exist_ok=True)
        items_path = index_dir / "items.jsonl"
        self._write_items_jsonl(chunk_output, items_path)
        return self._build_dense_faiss(
            chunk_output=chunk_output,
            dataset=dataset,
            index_dir=index_dir,
            items_path=items_path,
        )

    def try_load_reusable_output(
        self,
        *,
        chunk_output: Any,
        dataset: Any,
    ) -> dict[str, Any] | None:
        index_dir = self.resolve_index_dir()
        manifest_path = index_dir / "manifest.json"
        if not manifest_path.exists():
            return None

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return None

        if not self._manifest_matches(manifest, chunk_output):
            return None

        items_path = Path(str(manifest.get("items_path", "")))
        index_path = Path(str(manifest.get("index_path", "")))
        if not items_path.exists() or not index_path.exists():
            return None

        output: dict[str, Any] = {
            "backend": str(manifest.get("backend", "dense_faiss")),
            "index_dir": str(index_dir.resolve()),
            "index_path": str(index_path.resolve()),
            "items_path": str(items_path.resolve()),
            "manifest_path": str(manifest_path.resolve()),
            "metadata": {
                "dataset_name": self.dataset_name,
                "chunking_name": self.chunking_name,
                "run_name": self.run_name,
                "retrieval_backend": str(manifest.get("backend", "dense_faiss")),
                "retrieval_model": self._embedding_model_name(),
                "distance": self._distance_metric(),
                "top_k": self._top_k(),
                "reused_existing_index": True,
            },
        }

        run_path_value = manifest.get("run_path")
        if isinstance(run_path_value, str) and run_path_value.strip():
            run_path = Path(run_path_value)
            if run_path.exists():
                output["run_path"] = str(run_path.resolve())
                if dataset is not None and hasattr(dataset, "get_topics"):
                    topics = dataset.get_topics()
                    if not isinstance(topics, pd.DataFrame):
                        topics = pd.DataFrame(topics)
                    output["topics"] = topics
                if dataset is not None and hasattr(dataset, "get_qrels"):
                    output["qrels"] = dataset.get_qrels()

        return output

    def _build_dense_faiss(
        self,
        *,
        chunk_output: Any,
        dataset: Any,
        index_dir: Path,
        items_path: Path,
    ) -> dict[str, Any]:
        try:
            import faiss
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "FAISS is required for dense retrieval but is not installed."
            ) from exc

        chunks = self._materialize_chunks(chunk_output)
        chunk_texts = [chunk["text"] for chunk in chunks]
        if not chunk_texts:
            raise ValueError("Dense retrieval requires at least one chunk with non-empty text.")

        embedder = EmbedderFactory.create(self._embedding_model_name())
        doc_embeddings, doc_embedding_metadata = embedder.encode_texts(
            chunk_texts,
            self._dense_embedding_config(),
        )
        doc_matrix = np.asarray(doc_embeddings, dtype=np.float32)
        if doc_matrix.ndim != 2 or doc_matrix.shape[0] != len(chunks):
            raise ValueError("Dense retrieval chunk embeddings have invalid shape.")

        distance = self._distance_metric()
        faiss_index = self._build_faiss_index(faiss=faiss, matrix=doc_matrix, distance=distance)
        index_path = index_dir / "index.faiss"
        faiss.write_index(faiss_index, str(index_path))

        output: dict[str, Any] = {
            "backend": "dense_faiss",
            "index_dir": str(index_dir.resolve()),
            "index_path": str(index_path.resolve()),
            "items_path": str(items_path.resolve()),
            "metadata": {
                "dataset_name": self.dataset_name,
                "chunking_name": self.chunking_name,
                "run_name": self.run_name,
                "retrieval_backend": "dense_faiss",
                "retrieval_model": self._embedding_model_name(),
                "distance": distance,
                "embedding_dim": int(doc_matrix.shape[1]),
                "top_k": self._top_k(),
                "embedding_metadata": doc_embedding_metadata,
            },
        }

        if dataset is not None and hasattr(dataset, "get_topics"):
            topics = dataset.get_topics()
            if not isinstance(topics, pd.DataFrame):
                topics = pd.DataFrame(topics)
            query_texts = [str(value) for value in topics.get("query", pd.Series(dtype=str)).tolist()]
            query_embeddings, query_embedding_metadata = embedder.encode_texts(
                query_texts,
                self._dense_embedding_config(),
            )
            query_matrix = np.asarray(query_embeddings, dtype=np.float32)
            run = self._faiss_run_dataframe(
                faiss_index=faiss_index,
                query_matrix=query_matrix,
                topics=topics,
                chunks=chunks,
                top_k=self._top_k(),
                distance=distance,
            )
            output["topics"] = topics
            output["run"] = run
            output["metadata"]["query_embedding_metadata"] = query_embedding_metadata

            run_path = index_dir / "run.tsv"
            run.to_csv(run_path, sep="\t", index=False)
            output["run_path"] = str(run_path.resolve())

            if hasattr(dataset, "get_qrels"):
                output["qrels"] = dataset.get_qrels()

        manifest_path = index_dir / "manifest.json"
        manifest_payload = {
            "backend": "dense_faiss",
            "dataset_name": self.dataset_name,
            "chunking_name": self.chunking_name,
            "run_name": self.run_name,
            "index_dir": str(index_dir.resolve()),
            "index_path": str(index_path.resolve()),
            "retrieval_model": self._embedding_model_name(),
            "distance": distance,
            "items_path": str(items_path.resolve()),
            "has_run": "run" in output,
            "run_path": output.get("run_path"),
            "top_k": self._top_k(),
            "num_items": len(chunks),
            "chunk_signature": self._chunk_signature(chunks),
        }
        manifest_path.write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        output["manifest_path"] = str(manifest_path.resolve())
        return output

    def _manifest_matches(self, manifest: Any, chunk_output: Any) -> bool:
        if not isinstance(manifest, dict):
            return False
        chunks = self._materialize_chunks(chunk_output)
        if manifest.get("backend") != "dense_faiss":
            return False
        if manifest.get("dataset_name") != self.dataset_name:
            return False
        if manifest.get("chunking_name") != self.chunking_name:
            return False
        if manifest.get("run_name") != self.run_name:
            return False
        if manifest.get("retrieval_model") != self._embedding_model_name():
            return False
        if manifest.get("distance") != self._distance_metric():
            return False
        if int(manifest.get("top_k", 0) or 0) != self._top_k():
            return False
        if int(manifest.get("num_items", -1) or -1) != len(chunks):
            return False
        return manifest.get("chunk_signature") == self._chunk_signature(chunks)

    def resolve_index_dir(self) -> Path:
        explicit = self.retrieval_config.get("output_dir")
        if isinstance(explicit, str) and explicit.strip():
            return Path(explicit)
        return (
            Path("data/pt_indexes")
            / self._slugify(self.dataset_name)
            / self._slugify(self.chunking_name)
            / self._slugify(self.run_name)
        )

    def _resolve_inputs(self, inp: Any) -> tuple[Any, Any | None]:
        if isinstance(inp, dict):
            chunk_output = inp.get("chunk_output", inp)
            dataset = inp.get("dataset")
            return chunk_output, dataset
        return inp, None

    @staticmethod
    def _materialize_chunks(chunk_output: Any) -> list[dict[str, Any]]:
        if not isinstance(chunk_output, dict):
            raise TypeError("RetrievalTransformer expects a chunk_output dict.")
        chunks = chunk_output.get("chunks")
        if not isinstance(chunks, list):
            raise TypeError("chunk_output['chunks'] must be a list.")

        normalized: list[dict[str, Any]] = []
        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            docno = str(chunk.get("chunk_id", "")).strip()
            text = str(chunk.get("text", "")).strip()
            doc_id = str(chunk.get("doc_id", "")).strip()
            if not docno or not text:
                continue
            normalized.append(
                {
                "docno": docno,
                "text": text,
                "doc_id": doc_id,
                }
            )
        return normalized

    @staticmethod
    def _write_items_jsonl(chunk_output: Any, path: Path) -> None:
        if not isinstance(chunk_output, dict):
            raise TypeError("RetrievalTransformer expects a chunk_output dict.")
        chunks = chunk_output.get("chunks")
        if not isinstance(chunks, list):
            raise TypeError("chunk_output['chunks'] must be a list.")
        with path.open("w", encoding="utf-8") as f:
            for chunk in chunks:
                if not isinstance(chunk, dict):
                    continue
                row = {
                    "docno": str(chunk.get("chunk_id", "")).strip(),
                    "doc_id": str(chunk.get("doc_id", "")).strip(),
                    "text": str(chunk.get("text", "")).strip(),
                    "metadata": chunk.get("metadata", {}) if isinstance(chunk.get("metadata"), dict) else {},
                }
                if "sentences" in chunk:
                    row["sentences"] = chunk.get("sentences")
                if "start_unit_position" in chunk:
                    row["start_unit_position"] = chunk.get("start_unit_position")
                if "end_unit_position" in chunk:
                    row["end_unit_position"] = chunk.get("end_unit_position")
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _embedding_model_name(self) -> str:
        value = (
            self.retrieval_config.get("embedding_model")
            or self.retrieval_config.get("encoder")
        )
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                "Dense retrieval requires 'retrieval.embedding_model' (or 'retrieval.encoder')."
            )
        return value.strip()

    def _dense_embedding_config(self) -> dict[str, Any]:
        return {
            "batch_size": int(self.retrieval_config.get("embedding_batch_size", 32)),
            "normalize_embeddings": bool(self.retrieval_config.get("normalize", True)),
            "show_progress_bar": bool(self.retrieval_config.get("show_progress_bar", False)),
            "convert_to_numpy": True,
            "log_embedding_calls": bool(self.retrieval_config.get("log_embedding_calls", False)),
            "instruction": str(self.retrieval_config.get("instruction", "")),
        }

    def _distance_metric(self) -> str:
        value = self.retrieval_config.get("distance", "cosine")
        metric = str(value).strip().lower() or "cosine"
        supported = {"cosine", "ip", "inner_product", "l2", "euclidean"}
        if metric not in supported:
            raise ValueError(f"Unsupported dense retrieval distance '{metric}'.")
        return metric

    def _top_k(self) -> int:
        raw = self.retrieval_config.get("top_k", 100)
        value = int(raw)
        if value <= 0:
            raise ValueError("'retrieval.top_k' must be > 0.")
        return value

    @staticmethod
    def _build_faiss_index(*, faiss: Any, matrix: np.ndarray, distance: str) -> Any:
        working = np.asarray(matrix, dtype=np.float32)
        dim = int(working.shape[1])
        if distance in {"cosine", "ip", "inner_product"}:
            if distance == "cosine":
                faiss.normalize_L2(working)
            index = faiss.IndexFlatIP(dim)
        else:
            index = faiss.IndexFlatL2(dim)
        index.add(working)
        return index

    @staticmethod
    def _faiss_run_dataframe(
        *,
        faiss_index: Any,
        query_matrix: np.ndarray,
        topics: pd.DataFrame,
        chunks: list[dict[str, Any]],
        top_k: int,
        distance: str,
    ) -> pd.DataFrame:
        working = np.asarray(query_matrix, dtype=np.float32)
        if distance == "cosine":
            import faiss

            faiss.normalize_L2(working)
        scores, indices = faiss_index.search(working, min(top_k, len(chunks)))
        rows: list[dict[str, Any]] = []
        qids = [str(qid) for qid in topics["qid"].tolist()]
        queries = [str(query) for query in topics["query"].tolist()]

        for row_idx, (qid, query) in enumerate(zip(qids, queries)):
            for rank_idx, chunk_idx in enumerate(indices[row_idx].tolist(), start=1):
                if chunk_idx < 0 or chunk_idx >= len(chunks):
                    continue
                raw_score = float(scores[row_idx][rank_idx - 1])
                score = raw_score if distance in {"cosine", "ip", "inner_product"} else -raw_score
                rows.append(
                    {
                        "qid": qid,
                        "query": query,
                        "docno": chunks[chunk_idx]["docno"],
                        "rank": rank_idx,
                        "score": score,
                    }
                )
        return pd.DataFrame(rows, columns=["qid", "query", "docno", "rank", "score"])

    @staticmethod
    def _slugify(value: str) -> str:
        text = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value.strip())
        return text.strip("._") or "item"

    @staticmethod
    def _chunk_signature(chunks: list[dict[str, Any]]) -> str:
        digest = hashlib.sha1()
        for chunk in chunks:
            digest.update(str(chunk.get("docno", "")).encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(chunk.get("doc_id", "")).encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(chunk.get("text", "")).encode("utf-8"))
            digest.update(b"\n")
        return digest.hexdigest()
