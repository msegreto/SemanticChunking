from __future__ import annotations

from typing import Any

from src.chunking import _common


class FixedChunker:
    @property
    def chunking_type(self) -> str:
        return "fixed"

    def chunk(self, routed_output: Any, config: dict[str, Any]) -> dict[str, Any]:
        prepared = self.prepare_chunking_inputs(routed_output, config)
        cached_output = self.try_load_reusable_output(prepared)
        if cached_output is not None:
            return cached_output

        chunk_output = self.compute_chunk_output(prepared)
        self.save_output(chunk_output, prepared)
        return chunk_output

    def prepare_chunking_inputs(self, routed_output: Any, config: dict[str, Any]) -> dict[str, Any]:
        validated_config = self.validate_config(config)
        split_units = _common.validate_routed_output(routed_output)
        grouped_units = _common.group_split_units(split_units)
        run_dir = _common.build_run_dir(
            dataset=validated_config["dataset"],
            yaml_name=validated_config["yaml_name"],
        )
        return {
            "config": validated_config,
            "split_units": split_units,
            "grouped_units": grouped_units,
            "run_dir": run_dir,
        }

    def try_load_reusable_output(self, prepared: dict[str, Any]) -> dict[str, Any] | None:
        cached_run = _common.load_cached_run(
            run_dir=prepared["run_dir"],
            grouped_units=prepared["grouped_units"],
            config=prepared["config"],
            saved_payload_matches=self.saved_payload_matches,
        )
        if cached_run is None:
            return None
        all_chunks, documents = cached_run
        return _common.build_chunk_output(
            all_chunks=all_chunks,
            documents=documents,
            config=prepared["config"],
            run_dir=prepared["run_dir"],
            reused_existing_chunks=True,
            chunking_type=self.chunking_type,
            build_strategy_config_metadata=self.build_strategy_config_metadata,
        )

    def compute_chunk_output(self, prepared: dict[str, Any]) -> dict[str, Any]:
        all_chunks, documents = self.chunk_grouped_units(prepared["grouped_units"], prepared["config"])
        return _common.build_chunk_output(
            all_chunks=all_chunks,
            documents=documents,
            config=prepared["config"],
            run_dir=prepared["run_dir"],
            reused_existing_chunks=False,
            chunking_type=self.chunking_type,
            build_strategy_config_metadata=self.build_strategy_config_metadata,
        )

    def save_output(self, chunk_output: dict[str, Any], prepared: dict[str, Any]) -> None:
        if prepared["config"]["save_chunks"]:
            _common.save_run_chunks(
                run_dir=prepared["run_dir"],
                grouped_chunks=chunk_output["chunks"],
                config=prepared["config"],
                build_saved_payload_metadata=self.build_saved_payload_metadata,
            )

    def validate_config(self, config: Any) -> dict[str, Any]:
        return _common.validate_chunking_config(config, self.validate_strategy_config)

    def build_run_dir(self, dataset: str, yaml_name: str):
        return _common.build_run_dir(dataset, yaml_name)

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
            ordered_units = sorted(units, key=lambda item: item["position"])
            sentences = [unit["text"] for unit in ordered_units]
            chunks = self._build_chunks(
                doc_id=doc_id,
                sentences=sentences,
                n_chunks=config["n_chunks"],
                overlap=config["overlap_sentences"],
            )
            grouped_chunks.append((doc_id, chunks))
            all_chunks.extend(chunks)

        _, documents = _common.build_documents_from_grouped_chunks(
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
                    "start_unit_position": start_idx,
                    "end_unit_position": end_idx,
                    "position": position,
                    "metadata": {
                        "chunking_type": self.chunking_type,
                        "overlap_sentences": overlap,
                    },
                }
            )

            cursor += size

        return chunks
