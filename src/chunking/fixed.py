from __future__ import annotations

from typing import Any

from src.chunking.base import BaseChunker


class FixedChunker(BaseChunker):
    @property
    def chunking_type(self) -> str:
        return "fixed"

    def validate_strategy_config(self, config: dict[str, Any]) -> dict[str, Any]:
        n_chunks = config.get("n_chunks")
        if isinstance(n_chunks, bool) or not isinstance(n_chunks, int) or n_chunks <= 0:
            raise ValueError("Chunking config requires 'n_chunks' as a positive integer.")

        overlap = config.get("overlap_sentences", 0)
        if isinstance(overlap, bool) or not isinstance(overlap, int) or overlap < 0:
            raise ValueError(
                "Chunking config requires 'overlap_sentences' as an integer >= 0."
            )

        config["n_chunks"] = n_chunks
        config["overlap_sentences"] = overlap
        return config

    def chunk_grouped_units(
        self,
        grouped_units: dict[str, list[dict[str, Any]]],
        config: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        all_chunks: list[dict[str, Any]] = []
        grouped_chunks: list[tuple[str, list[dict[str, Any]]]] = []

        for doc_id, units in grouped_units.items():
            ordered_units = sorted(units, key=lambda item: item["sentence_idx"])
            sentences = [unit["text"] for unit in ordered_units]
            chunks = self._build_chunks(
                doc_id=doc_id,
                sentences=sentences,
                n_chunks=config["n_chunks"],
                overlap=config["overlap_sentences"],
            )
            grouped_chunks.append((doc_id, chunks))
            all_chunks.extend(chunks)

        _, documents = self.build_documents_from_grouped_chunks(
            grouped_units=grouped_units,
            grouped_chunks=grouped_chunks,
            reused=False,
        )
        return all_chunks, documents

    def saved_payload_matches(
        self,
        metadata: Any,
        doc_id: str,
        config: dict[str, Any],
    ) -> bool:
        if not isinstance(metadata, dict):
            return False

        return (
            metadata.get("chunking_type") == self.chunking_type
            and metadata.get("dataset") == config["dataset"]
            and metadata.get("doc_id") == doc_id
            and metadata.get("n_chunks") == config["n_chunks"]
            and metadata.get("overlap_sentences") == config["overlap_sentences"]
            and metadata.get("yaml_name") == config["yaml_name"]
        )

    def build_saved_payload_metadata(
        self,
        doc_id: str,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "chunking_type": self.chunking_type,
            "dataset": config["dataset"],
            "doc_id": doc_id,
            "n_chunks": config["n_chunks"],
            "overlap_sentences": config["overlap_sentences"],
            "yaml_name": config["yaml_name"],
        }

    def build_strategy_config_metadata(self, config: dict[str, Any]) -> dict[str, Any]:
        return {
            "n_chunks": config["n_chunks"],
            "overlap_sentences": config["overlap_sentences"],
        }

    def _build_chunks(
        self,
        doc_id: str,
        sentences: list[str],
        n_chunks: int,
        overlap: int,
    ) -> list[dict[str, Any]]:
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
                        "chunking_type": self.chunking_type,
                        "overlap_sentences": overlap,
                    },
                }
            )

            cursor += size

        return chunks
