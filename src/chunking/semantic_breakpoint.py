from __future__ import annotations

import json
from typing import Any

from langchain_experimental.text_splitter import (
    BREAKPOINT_DEFAULTS,
    SemanticChunker as LangChainSemanticChunker,
)
from langchain_huggingface import HuggingFaceEmbeddings

from src.chunking.semantic_base import BaseSemanticChunker
from src.chunking.semantic_utils import SUPPORTED_BREAKPOINT_THRESHOLD_TYPES


class SemanticBreakpointChunker(BaseSemanticChunker):
    def __init__(self) -> None:
        super().__init__()
        self._langchain_embeddings: HuggingFaceEmbeddings | None = None
        self._langchain_embeddings_key: tuple[Any, ...] | None = None
        self._langchain_chunker: LangChainSemanticChunker | None = None
        self._langchain_chunker_key: tuple[Any, ...] | None = None

    @property
    def chunking_type(self) -> str:
        return "semantic_breakpoint"

    def chunk(self, routed_output: Any, config: dict) -> Any:
        print(f"[CHUNKING] semantic breakpoint chunking with config={config}")
        return super().chunk(routed_output, config)

    def validate_semantic_method_config(self, config: dict[str, Any]) -> dict[str, Any]:
        threshold_type = config.get("threshold_type", "percentile")
        if not isinstance(threshold_type, str) or not threshold_type.strip():
            raise ValueError(
                "Semantic breakpoint config requires 'threshold_type' as a non-empty string."
            )
        threshold_type = threshold_type.strip().lower()
        if threshold_type not in SUPPORTED_BREAKPOINT_THRESHOLD_TYPES:
            available = ", ".join(sorted(SUPPORTED_BREAKPOINT_THRESHOLD_TYPES))
            raise ValueError(
                "Semantic breakpoint config requires a supported 'threshold_type'. "
                f"Available: {available}. Got '{threshold_type}'."
            )

        threshold_value = config.get("threshold_value")
        if threshold_value is not None:
            if isinstance(threshold_value, bool) or not isinstance(threshold_value, (int, float)):
                raise ValueError(
                    "Semantic breakpoint config requires 'threshold_value' to be numeric "
                    "or omitted to use the LangChain default."
                )
            threshold_value = float(threshold_value)
            if threshold_type in {"percentile", "gradient"} and not 0.0 <= threshold_value <= 100.0:
                raise ValueError(
                    f"Semantic breakpoint '{threshold_type}' requires 'threshold_value' in [0, 100]."
                )
            if threshold_type in {"standard_deviation", "interquartile"} and threshold_value < 0.0:
                raise ValueError(
                    f"Semantic breakpoint '{threshold_type}' requires 'threshold_value' >= 0."
                )

        buffer_size = config.get("buffer_size", 1)
        if isinstance(buffer_size, bool) or not isinstance(buffer_size, int) or buffer_size < 0:
            raise ValueError(
                "Semantic breakpoint config requires 'buffer_size' as an integer >= 0."
            )

        min_chunk_size = config.get("min_chunk_size")
        if min_chunk_size is not None:
            if isinstance(min_chunk_size, bool) or not isinstance(min_chunk_size, int) or min_chunk_size <= 0:
                raise ValueError(
                    "Semantic breakpoint config requires 'min_chunk_size' to be a positive "
                    "integer or omitted."
                )

        number_of_chunks = config.get("number_of_chunks")
        if number_of_chunks is not None:
            if isinstance(number_of_chunks, bool) or not isinstance(number_of_chunks, int) or number_of_chunks <= 0:
                raise ValueError(
                    "Semantic breakpoint config requires 'number_of_chunks' to be a positive "
                    "integer or omitted."
                )

        embedding_model_kwargs = config.get("embedding_model_kwargs", {})
        if not isinstance(embedding_model_kwargs, dict):
            raise TypeError(
                "Semantic breakpoint config requires 'embedding_model_kwargs' as a dict."
            )

        embedding_encode_kwargs = config.get("embedding_encode_kwargs", {})
        if not isinstance(embedding_encode_kwargs, dict):
            raise TypeError(
                "Semantic breakpoint config requires 'embedding_encode_kwargs' as a dict."
            )

        sentence_split_regex = config.get("sentence_split_regex", r"(?<=[.?!])\s+")
        if not isinstance(sentence_split_regex, str) or not sentence_split_regex.strip():
            raise ValueError(
                "Semantic breakpoint config requires 'sentence_split_regex' as a non-empty string."
            )

        prepared = dict(config)
        prepared["threshold_type"] = threshold_type
        prepared["threshold_value"] = threshold_value
        prepared["buffer_size"] = buffer_size
        prepared["min_chunk_size"] = min_chunk_size
        prepared["number_of_chunks"] = number_of_chunks
        prepared["embedding_model_kwargs"] = dict(embedding_model_kwargs)
        prepared["embedding_encode_kwargs"] = dict(embedding_encode_kwargs)
        prepared["sentence_split_regex"] = sentence_split_regex
        prepared["paper_method_family"] = "breakpoint"
        prepared["implementation_status"] = "implemented"
        prepared["implementation_backend"] = "langchain"
        return prepared

    def chunk_grouped_units(
        self,
        grouped_units: dict[str, list[dict[str, Any]]],
        config: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        all_chunks: list[dict[str, Any]] = []
        grouped_chunks: list[tuple[str, list[dict[str, Any]]]] = []

        for doc_id, units in grouped_units.items():
            ordered_units = self.ordered_units(units)
            chunks = self._build_breakpoint_chunks(
                doc_id=doc_id,
                units=ordered_units,
                config=config,
            )
            grouped_chunks.append((doc_id, chunks))
            all_chunks.extend(chunks)

        _, documents = self.build_documents_from_grouped_chunks(
            grouped_units=grouped_units,
            grouped_chunks=grouped_chunks,
            reused=False,
        )
        return all_chunks, documents

    def build_semantic_method_metadata(self, config: dict[str, Any]) -> dict[str, Any]:
        return {
            "paper_method_family": config["paper_method_family"],
            "threshold_type": config["threshold_type"],
            "threshold_value": config["threshold_value"],
            "buffer_size": config["buffer_size"],
            "min_chunk_size": config["min_chunk_size"],
            "number_of_chunks": config["number_of_chunks"],
            "sentence_split_regex": config["sentence_split_regex"],
            "embedding_model_kwargs": dict(config["embedding_model_kwargs"]),
            "embedding_encode_kwargs": dict(config["embedding_encode_kwargs"]),
            "implementation_status": config["implementation_status"],
            "implementation_backend": config["implementation_backend"],
        }

    def semantic_method_metadata_matches(
        self,
        metadata: dict[str, Any],
        config: dict[str, Any],
    ) -> bool:
        return (
            metadata.get("paper_method_family") == config["paper_method_family"]
            and metadata.get("threshold_type") == config["threshold_type"]
            and metadata.get("threshold_value") == config["threshold_value"]
            and metadata.get("buffer_size") == config["buffer_size"]
            and metadata.get("min_chunk_size") == config["min_chunk_size"]
            and metadata.get("number_of_chunks") == config["number_of_chunks"]
            and metadata.get("sentence_split_regex") == config["sentence_split_regex"]
            and metadata.get("embedding_model_kwargs") == config["embedding_model_kwargs"]
            and metadata.get("embedding_encode_kwargs") == config["embedding_encode_kwargs"]
            and metadata.get("implementation_status") == config["implementation_status"]
            and metadata.get("implementation_backend") == config["implementation_backend"]
        )

    def _build_breakpoint_chunks(
        self,
        doc_id: str,
        units: list[dict[str, Any]],
        config: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if not units:
            return []

        if len(units) == 1:
            return [
                self._build_chunk_from_span(
                    doc_id=doc_id,
                    units=units,
                    start_idx=0,
                    end_idx=0,
                    position=0,
                    config=config,
                    boundary_score=None,
                    threshold=None,
                )
            ]

        sentences = [unit["text"].strip() for unit in units]
        chunker = self._get_langchain_chunker(config)
        threshold, breakpoint_scores = self._compute_langchain_breakpoints(
            chunker=chunker,
            sentences=sentences,
        )

        chunks: list[dict[str, Any]] = []
        start_idx = 0

        for boundary_idx in breakpoint_scores["indices_above_threshold"]:
            candidate_text = self._join_unit_texts(units[start_idx:boundary_idx + 1])
            if config["min_chunk_size"] is not None and len(candidate_text) < config["min_chunk_size"]:
                continue

            chunks.append(
                self._build_chunk_from_span(
                    doc_id=doc_id,
                    units=units,
                    start_idx=start_idx,
                    end_idx=boundary_idx,
                    position=len(chunks),
                    config=config,
                    boundary_score=breakpoint_scores["breakpoint_array"][boundary_idx],
                    threshold=threshold,
                )
            )
            start_idx = boundary_idx + 1

        if start_idx < len(units):
            chunks.append(
                self._build_chunk_from_span(
                    doc_id=doc_id,
                    units=units,
                    start_idx=start_idx,
                    end_idx=len(units) - 1,
                    position=len(chunks),
                    config=config,
                    boundary_score=None,
                    threshold=threshold,
                )
            )

        return chunks

    def _compute_langchain_breakpoints(
        self,
        chunker: LangChainSemanticChunker,
        sentences: list[str],
    ) -> tuple[float, dict[str, Any]]:
        if len(sentences) == 1:
            return float("inf"), {"breakpoint_array": [], "indices_above_threshold": []}

        if chunker.breakpoint_threshold_type == "gradient" and len(sentences) == 2:
            return float("inf"), {"breakpoint_array": [], "indices_above_threshold": []}

        distances, _ = chunker._calculate_sentence_distances(sentences)
        if chunker.number_of_chunks is not None:
            threshold = float(chunker._threshold_from_clusters(distances))
            breakpoint_array = distances
        else:
            threshold, breakpoint_array = chunker._calculate_breakpoint_threshold(distances)
            threshold = float(threshold)

        indices_above_threshold = [
            idx for idx, value in enumerate(breakpoint_array) if value > threshold
        ]
        return threshold, {
            "distances": distances,
            "breakpoint_array": breakpoint_array,
            "indices_above_threshold": indices_above_threshold,
        }

    def _get_langchain_chunker(self, config: dict[str, Any]) -> LangChainSemanticChunker:
        key = (
            config["embedding_model"],
            json.dumps(config["embedding_model_kwargs"], sort_keys=True),
            json.dumps(config["embedding_encode_kwargs"], sort_keys=True),
            config["embedding_batch_size"],
            config["threshold_type"],
            config["threshold_value"],
            config["number_of_chunks"],
            config["buffer_size"],
            config["sentence_split_regex"],
            config["min_chunk_size"],
        )
        if self._langchain_chunker is None or self._langchain_chunker_key != key:
            self._langchain_chunker = LangChainSemanticChunker(
                embeddings=self._get_langchain_embeddings(config),
                buffer_size=config["buffer_size"],
                add_start_index=False,
                breakpoint_threshold_type=config["threshold_type"],
                breakpoint_threshold_amount=config["threshold_value"],
                number_of_chunks=config["number_of_chunks"],
                sentence_split_regex=config["sentence_split_regex"],
                min_chunk_size=config["min_chunk_size"],
            )
            self._langchain_chunker_key = key
        return self._langchain_chunker

    def _get_langchain_embeddings(self, config: dict[str, Any]) -> HuggingFaceEmbeddings:
        key = (
            config["embedding_model"],
            json.dumps(config["embedding_model_kwargs"], sort_keys=True),
            json.dumps(config["embedding_encode_kwargs"], sort_keys=True),
            config["embedding_batch_size"],
        )
        if self._langchain_embeddings is None or self._langchain_embeddings_key != key:
            encode_kwargs = {
                "batch_size": config["embedding_batch_size"],
                "normalize_embeddings": True,
                **config["embedding_encode_kwargs"],
            }
            self._langchain_embeddings = HuggingFaceEmbeddings(
                model_name=config["embedding_model"],
                model_kwargs=dict(config["embedding_model_kwargs"]),
                encode_kwargs=encode_kwargs,
                show_progress=False,
            )
            self._langchain_embeddings_key = key
        return self._langchain_embeddings

    def _build_chunk_from_span(
        self,
        doc_id: str,
        units: list[dict[str, Any]],
        start_idx: int,
        end_idx: int,
        position: int,
        config: dict[str, Any],
        boundary_score: float | None,
        threshold: float | None,
    ) -> dict[str, Any]:
        span = units[start_idx:end_idx + 1]
        first_unit = span[0]
        last_unit = span[-1]
        sentences = [unit["text"] for unit in span]

        return {
            "chunk_id": f"{doc_id}::chunk_{position}",
            "doc_id": doc_id,
            "text": self._join_unit_texts(span),
            "sentences": sentences,
            "start_sentence_idx": first_unit["sentence_idx"],
            "end_sentence_idx": last_unit["sentence_idx"],
            "position": position,
            "metadata": {
                "chunking_type": self.chunking_type,
                "embedding_model": config["embedding_model"],
                "similarity_metric": config["similarity_metric"],
                "threshold_type": config["threshold_type"],
                "threshold_value": config["threshold_value"],
                "threshold_value_effective": (
                    config["threshold_value"]
                    if config["threshold_value"] is not None
                    else BREAKPOINT_DEFAULTS[config["threshold_type"]]
                ),
                "buffer_size": config["buffer_size"],
                "min_chunk_size": config["min_chunk_size"],
                "number_of_chunks": config["number_of_chunks"],
                "implementation_backend": config["implementation_backend"],
                "boundary_score": boundary_score,
                "threshold_used": threshold,
            },
        }

    def _join_unit_texts(self, units: list[dict[str, Any]]) -> str:
        return " ".join(unit["text"].strip() for unit in units if unit["text"].strip())
