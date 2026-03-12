from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any
import json

from src.chunking.base import BaseChunker


class FixedChunker(BaseChunker):
    def chunk(self, routed_output: Any, config: dict) -> Any:
        print(f"[CHUNKING] fixed chunking with config={config}")

        split_units = routed_output["split_units"]
        dataset = config["dataset"]
        n_chunks = config["n_chunks"]
        overlap = config.get("overlap_sentences", 0)
        save_chunks = config.get("save_chunks", True)

        grouped = defaultdict(list)
        for unit in split_units:
            grouped[unit["doc_id"]].append(unit)

        all_chunks = []
        documents = []

        for doc_id, units in grouped.items():
            units = sorted(units, key=lambda x: x["sentence_idx"])
            sentences = [u["text"] for u in units]

            chunks = self._build_chunks(doc_id, sentences, n_chunks, overlap)
            all_chunks.extend(chunks)

            if save_chunks:
                self._save_chunks(dataset, doc_id, chunks)

            documents.append(
                {
                    "doc_id": doc_id,
                    "num_input_sentences": len(sentences),
                    "num_chunks": len(chunks),
                }
            )

        return {
            "chunks": all_chunks,
            "documents": documents,
            "metadata": {
                "chunking_type": "fixed",
                "num_documents": len(documents),
                "total_chunks": len(all_chunks),
                "config_used": {
                    "dataset": dataset,
                    "n_chunks": n_chunks,
                    "overlap_sentences": overlap,
                    "save_chunks": save_chunks,
                },
            },
        }

    def _build_chunks(
        self,
        doc_id: str,
        sentences: list[str],
        n_chunks: int,
        overlap: int,
    ) -> list[dict]:
        if not sentences:
            return []

        actual_n_chunks = min(n_chunks, len(sentences))
        base_size = len(sentences) // actual_n_chunks
        remainder = len(sentences) % actual_n_chunks

        chunk_sizes = [
            base_size + (1 if i < remainder else 0)
            for i in range(actual_n_chunks)
        ]

        chunks = []
        cursor = 0

        for position, size in enumerate(chunk_sizes):
            base_start = cursor
            base_end = cursor + size - 1

            start_idx = max(0, base_start - overlap)
            end_idx = min(len(sentences) - 1, base_end + overlap)

            chunk_sentences = sentences[start_idx:end_idx + 1]

            chunks.append(
                {
                    "chunk_id": f"{doc_id}::chunk_{position}",
                    "doc_id": doc_id,
                    "text": " ".join(s.strip() for s in chunk_sentences if s.strip()),
                    "sentences": chunk_sentences,
                    "start_sentence_idx": start_idx,
                    "end_sentence_idx": end_idx,
                    "position": position,
                    "metadata": {
                        "chunking_type": "fixed",
                        "overlap_sentences": overlap,
                    },
                }
            )

            cursor += size

        return chunks

    def _save_chunks(self, dataset: str, doc_id: str, chunks: list[dict]) -> None:
        save_dir = Path("data/chunks") / f"{dataset}_fixed"
        save_dir.mkdir(parents=True, exist_ok=True)

        safe_doc_id = doc_id.replace("/", "_")
        path = save_dir / f"{safe_doc_id}.json"

        with open(path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)