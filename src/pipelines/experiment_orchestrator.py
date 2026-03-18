from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml
from tqdm.auto import tqdm

from src.chunking.factory import ChunkerFactory
from src.datasets.factory import DatasetFactory
from src.embeddings.factory import EmbedderFactory
from src.evaluation.intrinsic.factory import IntrinsicEvaluatorFactory
from src.retrieval.factory import RetrieverFactory
from src.routing.factory import RouterFactory
from src.splitting.factory import SplitterFactory



@dataclass
class ExperimentArtifacts:
    dataset_output: Any = None
    split_output: Any = None
    routed_output: Any = None
    chunk_output: Any = None
    intrinsic_output: Any = None
    embedding_output: Any = None
    index_output: Any = None


@dataclass
class StreamingWindowPaths:
    split_tmp: Path
    chunks_tmp: Path


class ExperimentOrchestrator:
    def __init__(self, config: Dict[str, Any], config_path: Optional[Path] = None) -> None:
        self.config = config
        self.config_path = config_path
        self.artifacts = ExperimentArtifacts()

    @classmethod
    def from_yaml(cls, config_path: Path) -> "ExperimentOrchestrator":
        with config_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return cls(config=config, config_path=config_path)

    def run(self) -> None:
        print("[INFO] Starting experiment pipeline...")

        dataset_output = self._run_dataset_processing()
        self.artifacts.dataset_output = dataset_output

        # Pipeline is streaming-only: always execute the incremental path.
        self._run_streaming_pipeline(dataset_output)
        print("[INFO] Pipeline completed successfully.")

    def _run_streaming_pipeline(self, dataset_output: Any) -> None:
        split_cfg = self.config["split"]
        split_type = split_cfg["type"]
        if split_type != "sentence":
            raise ValueError("Streaming mode currently supports only split.type='sentence'.")

        router_cfg = self.config.get("router", {})
        router_enabled = bool(router_cfg.get("enabled", False))
        router = None
        if router_enabled:
            router_name = str(router_cfg.get("name", "default"))
            print(f"[INFO] Running router in streaming mode: {router_name}")
            router = RouterFactory.create(router_name)

        splitter = SplitterFactory.create(split_type)
        if not hasattr(splitter, "build_streaming_components") or not hasattr(splitter, "split_document_streaming"):
            raise ValueError(f"Splitter '{split_type}' does not support incremental streaming mode.")

        split_components = splitter.build_streaming_components(split_cfg)
        split_output_path = splitter.resolve_output_path(split_cfg)
        split_save_output = bool(split_cfg.get("save_output", False))
        split_path_obj = split_output_path if (split_save_output and split_output_path is not None) else None

        chunking_cfg = dict(self.config["chunking"])
        embedding_cfg = self.config.get("embedding", {})
        if isinstance(embedding_cfg.get("name"), str) and embedding_cfg["name"].strip():
            chunking_cfg.setdefault("embedding_model", embedding_cfg["name"].strip())
        if isinstance(embedding_cfg.get("show_progress_bar"), bool):
            chunking_cfg.setdefault(
                "show_embedding_progress",
                embedding_cfg["show_progress_bar"],
            )
        if self.config_path is not None:
            chunking_cfg.setdefault("yaml_name", self.config_path.name)
        else:
            chunking_cfg.setdefault("yaml_name", self.config.get("experiment_name"))

        chunker = ChunkerFactory.create(chunking_cfg["type"])
        validated_chunk_cfg = chunker.validate_config(chunking_cfg)
        run_dir = chunker.build_run_dir(
            dataset=validated_chunk_cfg["dataset"],
            yaml_name=validated_chunk_cfg["yaml_name"],
        )
        self.artifacts.routed_output = {
            "metadata": {
                "streaming": True,
                "router_enabled": router_enabled,
                "router_name": router_cfg.get("name", "default") if router_enabled else "no-routing",
            }
        }

        intrinsic_cfg = self.config.get("evaluation", {})
        intrinsic_enabled = intrinsic_cfg.get("intrinsic", True)
        if intrinsic_enabled:
            print("[INFO] Running intrinsic evaluation (streaming, incremental aggregates)...")

        retrieval_cfg = self.config.get("retrieval", {})
        window_docs = self._streaming_docs_per_run()
        run_until_complete = self._streaming_run_until_complete()
        stream_state_path = self._resolve_stream_state_path(retrieval_cfg)
        stream_state = self._load_stream_state(stream_state_path)
        split_state_path = self._resolve_split_state_path(split_path_obj)
        split_state = self._load_split_state(split_state_path)
        retrieval_enabled = bool(retrieval_cfg.get("enabled", True))
        if retrieval_enabled:
            # Streaming progress is owned by index checkpoint for the active retrieval/chunking config.
            # Do not inherit doc progress from split_state by default, otherwise a different index
            # config can appear complete without a corresponding index checkpoint.
            next_doc_idx = int(stream_state.get("next_doc_idx", 0))
            next_unit_id = int(stream_state.get("next_unit_id", 0))
        else:
            next_doc_idx = int(stream_state.get("next_doc_idx", split_state.get("next_doc_idx", 0)))
            next_unit_id = int(stream_state.get("next_unit_id", split_state.get("next_unit_id", 0)))

        expected_docs = dataset_output.get("metadata", {}).get("num_documents")
        if not isinstance(expected_docs, int) or expected_docs < 0:
            expected_docs = 0

        split_completed = bool(split_state.get("completed", False))
        split_expected_docs = split_state.get("expected_docs")
        split_expected_matches = not isinstance(split_expected_docs, int) or split_expected_docs == expected_docs
        reuse_saved_split_units = (
            split_save_output
            and split_path_obj is not None
            and split_path_obj.exists()
            and split_completed
            and split_expected_matches
            and self._should_allow_reuse("split")
        )
        if reuse_saved_split_units:
            print(
                "[INFO] Split shortcut activated: reusing saved split units from "
                f"{split_path_obj}."
            )
        elif split_completed and not split_expected_matches:
            print(
                "[INFO] split_state expected_docs mismatch with current dataset; "
                "saved split units will not be reused."
            )

        total_docs = int(stream_state.get("total_docs_processed", 0))
        total_chunks = int(stream_state.get("total_chunks_processed", 0))
        total_index_items = int(stream_state.get("total_index_items", 0))
        if retrieval_enabled:
            total_units = int(stream_state.get("total_units_processed", 0))
        else:
            total_units = int(stream_state.get("total_units_processed", split_state.get("total_units_processed", 0)))

        if split_save_output and split_path_obj is not None and next_doc_idx == 0 and not reuse_saved_split_units:
            split_path_obj.parent.mkdir(parents=True, exist_ok=True)
            split_path_obj.write_text("", encoding="utf-8")

        split_output = {
            "split_units_path": str(split_path_obj) if split_path_obj else None,
            "documents": [],
            "metadata": {
                "split_type": "sentence",
                "spacy_model": split_components["model_name"],
                "num_documents": expected_docs,
                "num_units": total_units,
                "source_metadata": dataset_output.get("metadata", {}),
                "reused_existing_split": bool(reuse_saved_split_units or next_doc_idx > 0),
                "split_path": str(split_path_obj) if split_path_obj else None,
                "streaming": True,
                "incremental_by_window": True,
            },
        }
        self.artifacts.split_output = split_output

        stream_embedding_cfg = dict(embedding_cfg)
        stream_embedding_cfg["show_progress_bar"] = False
        stream_embedding_cfg["log_embedding_calls"] = False

        doc_iter = None
        split_units_iter = None
        next_saved_units_pair = None
        if reuse_saved_split_units and split_path_obj is not None:
            doc_iter = enumerate(self._iter_documents_from_normalized(dataset_output))
            split_units_iter = iter(self._iter_split_units_grouped_by_doc(split_path_obj))
            next_saved_units_pair = self._advance_doc_and_split_iterators(
                doc_iter=doc_iter,
                split_units_iter=split_units_iter,
                next_doc_idx=next_doc_idx,
            )
        else:
            doc_iter = enumerate(self._iter_documents_from_normalized(dataset_output))
            self._advance_document_iterator(doc_iter, next_doc_idx)

        final_index_output = None
        final_intrinsic_output = None

        while True:
            if next_doc_idx >= expected_docs:
                stream_complete = True
                break

            end_doc_idx_exclusive = (
                min(next_doc_idx + window_docs, expected_docs) if window_docs > 0 else expected_docs
            )
            print(
                "[INFO] Streaming window: "
                f"start_doc={next_doc_idx}, end_doc={end_doc_idx_exclusive}, total_docs={expected_docs}"
            )

            intrinsic_runtime = self._build_intrinsic_runtime_for_window(
                intrinsic_enabled=intrinsic_enabled,
                intrinsic_cfg=intrinsic_cfg,
                stream_state=stream_state,
            )
            retriever, embedder = self._start_window_retrieval(
                retrieval_cfg=retrieval_cfg,
                next_doc_idx=next_doc_idx,
            )
            window_paths = self._prepare_window_paths(
                stream_state_path=stream_state_path,
                next_doc_idx=next_doc_idx,
                end_doc_idx_exclusive=end_doc_idx_exclusive,
            )

            if split_units_iter is not None:
                window_doc_ids, next_unit_id, total_docs, total_units, next_saved_units_pair = self._run_split_window_from_saved_split(
                    doc_iter=doc_iter,
                    split_units_iter=split_units_iter,
                    next_saved_units_pair=next_saved_units_pair,
                    next_unit_id=next_unit_id,
                    window_paths=window_paths,
                    next_doc_idx=next_doc_idx,
                    end_doc_idx_exclusive=end_doc_idx_exclusive,
                    total_docs=total_docs,
                    total_units=total_units,
                )
            else:
                window_doc_ids, next_unit_id, total_docs, total_units = self._run_split_window(
                    doc_iter=doc_iter,
                    splitter=splitter,
                    split_components=split_components,
                    next_unit_id=next_unit_id,
                    split_save_output=split_save_output,
                    split_path_obj=split_path_obj,
                    window_paths=window_paths,
                    next_doc_idx=next_doc_idx,
                    end_doc_idx_exclusive=end_doc_idx_exclusive,
                    total_docs=total_docs,
                    total_units=total_units,
                )

            current_window_docs = len(window_doc_ids)
            window_chunk_count, total_chunks = self._run_chunking_window(
                chunker=chunker,
                validated_chunk_cfg=validated_chunk_cfg,
                run_dir=run_dir,
                window_paths=window_paths,
                window_doc_ids=window_doc_ids,
                router=router,
                router_cfg=router_cfg,
                split_type=split_type,
                split_components=split_components,
                intrinsic_runtime=intrinsic_runtime,
                total_chunks=total_chunks,
            )

            total_index_items = self._run_indexing_window(
                window_paths=window_paths,
                retriever=retriever,
                embedder=embedder,
                stream_embedding_cfg=stream_embedding_cfg,
                window_chunk_count=window_chunk_count,
                total_index_items=total_index_items,
            )

            if retriever is not None:
                final_index_output = retriever.finalize_incremental_index()
                self.artifacts.index_output = final_index_output

            if intrinsic_runtime is not None:
                final_intrinsic_output = self._finalize_streaming_intrinsic_runtime(intrinsic_runtime, intrinsic_cfg)
                self.artifacts.intrinsic_output = final_intrinsic_output
                print(f"[INFO] Intrinsic evaluation completed (streaming): {final_intrinsic_output.get('metrics', {})}")
            else:
                self.artifacts.intrinsic_output = None

            self.artifacts.embedding_output = None
            next_doc_idx = next_doc_idx + current_window_docs
            stream_complete = next_doc_idx >= expected_docs
            stream_state = {
                "next_doc_idx": next_doc_idx,
                "next_unit_id": next_unit_id,
                "expected_docs": expected_docs,
                "total_docs_processed": total_docs,
                "total_units_processed": total_units,
                "total_chunks_processed": total_chunks,
                "total_index_items": total_index_items,
                "window_docs": window_docs,
                "completed": stream_complete,
                "intrinsic_runtime": self._serialize_intrinsic_runtime_for_state(intrinsic_runtime),
            }
            self._save_stream_state(stream_state_path, stream_state)
            self._save_split_state(
                split_state_path,
                {
                    "next_doc_idx": next_doc_idx,
                    "next_unit_id": next_unit_id,
                    "expected_docs": expected_docs,
                    "total_units_processed": total_units,
                    "window_docs": window_docs,
                    "completed": stream_complete,
                    "split_type": split_type,
                    "split_model": split_components.get("model_name"),
                    "split_path": str(split_path_obj) if split_path_obj else None,
                },
            )

            if stream_complete:
                break
            print(
                "[INFO] Streaming window completed. "
                f"Progress: next_doc_idx={next_doc_idx}/{expected_docs}."
            )
            if not run_until_complete:
                print("[INFO] Stopping after one window (streaming_run_until_complete=false).")
                break

        chunk_output = {
            "chunks": [],
            "documents": [],
            "metadata": {
                "chunking_type": validated_chunk_cfg["type"],
                "num_documents": total_docs,
                "total_units": total_units,
                "total_chunks": total_chunks,
                "reused_existing_chunks": False,
                "streaming": True,
                "split_path": str(split_path_obj) if split_path_obj else None,
            },
        }
        self.artifacts.chunk_output = chunk_output
        self.artifacts.intrinsic_output = final_intrinsic_output

        if retrieval_cfg.get("enabled", True) and stream_complete:
            if final_index_output is None:
                final_index_output = self._find_reusable_index(chunk_output)
                self.artifacts.index_output = final_index_output

            if final_index_output is not None:
                self._run_extrinsic_evaluation(
                    chunk_output=chunk_output,
                    index_output=final_index_output,
                )
            else:
                print(
                    "[INFO] Streaming marked complete but no reusable index was found. "
                    "Skipping extrinsic evaluation."
                )
        elif retrieval_cfg.get("enabled", True):
            print(
                "[INFO] Streaming not complete yet. "
                f"Progress: next_doc_idx={next_doc_idx}/{expected_docs}. "
                "Extrinsic evaluation deferred until completion."
            )

    @staticmethod
    def _advance_document_iterator(doc_iter: Any, next_doc_idx: int) -> None:
        while next_doc_idx > 0:
            try:
                doc_idx, _doc = next(doc_iter)
            except StopIteration:
                break
            if doc_idx + 1 >= next_doc_idx:
                break

    @staticmethod
    def _advance_doc_and_split_iterators(
        *,
        doc_iter: Any,
        split_units_iter: Any,
        next_doc_idx: int,
    ) -> tuple[str, list[dict[str, Any]]] | None:
        next_saved_units_pair = next(split_units_iter, None)
        advanced_docs = 0
        while advanced_docs < next_doc_idx:
            try:
                _doc_idx, doc = next(doc_iter)
            except StopIteration:
                break
            doc_id = str(doc.get("id", ""))
            if not doc_id:
                continue
            if next_saved_units_pair is not None and str(next_saved_units_pair[0]) == doc_id:
                next_saved_units_pair = next(split_units_iter, None)
            advanced_docs += 1
        return next_saved_units_pair

    def _build_intrinsic_runtime_for_window(
        self,
        *,
        intrinsic_enabled: bool,
        intrinsic_cfg: dict[str, Any],
        stream_state: dict[str, Any],
    ) -> dict[str, Any] | None:
        if not intrinsic_enabled:
            return None

        intrinsic_runtime = self._init_streaming_intrinsic_runtime(intrinsic_cfg)
        persisted_intrinsic = stream_state.get("intrinsic_runtime", {})
        if isinstance(persisted_intrinsic, dict):
            intrinsic_runtime["bc_sum"] += float(persisted_intrinsic.get("bc_sum", 0.0))
            intrinsic_runtime["bc_count"] += int(persisted_intrinsic.get("bc_count", 0))
            intrinsic_runtime["csc_sum"] += float(persisted_intrinsic.get("csc_sum", 0.0))
            intrinsic_runtime["csc_count"] += int(persisted_intrinsic.get("csc_count", 0))
            intrinsic_runtime["csi_sum"] += float(persisted_intrinsic.get("csi_sum", 0.0))
            intrinsic_runtime["csi_count"] += int(persisted_intrinsic.get("csi_count", 0))
            intrinsic_runtime["num_documents"] += int(persisted_intrinsic.get("num_documents", 0))
            intrinsic_runtime["num_chunks"] += int(persisted_intrinsic.get("num_chunks", 0))
        return intrinsic_runtime

    def _start_window_retrieval(
        self,
        *,
        retrieval_cfg: dict[str, Any],
        next_doc_idx: int,
    ) -> tuple[Any | None, Any | None]:
        if not retrieval_cfg.get("enabled", True):
            return None, None

        retriever_name = retrieval_cfg.get("name", "faiss")
        print(f"[INFO] Building retrieval index (streaming): {retriever_name}")
        retriever = RetrieverFactory.create(retriever_name)
        retrieval_cfg_for_stream = dict(retrieval_cfg)
        retrieval_cfg_for_stream["output_dir"] = str(self._resolve_retrieval_output_dir(retrieval_cfg))
        retrieval_cfg_for_stream["_stream_resume"] = next_doc_idx > 0
        retriever.start_incremental_index(retrieval_cfg_for_stream, self.config)
        embedder = EmbedderFactory.create(self.config["embedding"]["name"])
        return retriever, embedder

    @staticmethod
    def _prepare_window_paths(
        *,
        stream_state_path: Path,
        next_doc_idx: int,
        end_doc_idx_exclusive: int,
    ) -> StreamingWindowPaths:
        temp_dir = stream_state_path.parent / ".stream_tmp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        split_tmp = temp_dir / f"split_{next_doc_idx}_{end_doc_idx_exclusive}.jsonl"
        chunks_tmp = temp_dir / f"chunks_{next_doc_idx}_{end_doc_idx_exclusive}.jsonl"
        if split_tmp.exists():
            split_tmp.unlink()
        if chunks_tmp.exists():
            chunks_tmp.unlink()
        return StreamingWindowPaths(split_tmp=split_tmp, chunks_tmp=chunks_tmp)

    def _run_split_window(
        self,
        *,
        doc_iter: Any,
        splitter: Any,
        split_components: dict[str, Any],
        next_unit_id: int,
        split_save_output: bool,
        split_path_obj: Path | None,
        window_paths: StreamingWindowPaths,
        next_doc_idx: int,
        end_doc_idx_exclusive: int,
        total_docs: int,
        total_units: int,
    ) -> tuple[list[str], int, int, int]:
        window_target_docs = max(0, end_doc_idx_exclusive - next_doc_idx)
        window_doc_ids: list[str] = []
        split_progress = tqdm(
            total=window_target_docs,
            desc="Split (streaming)",
            unit="doc",
            dynamic_ncols=True,
            miniters=1,
        )
        try:
            while len(window_doc_ids) < window_target_docs:
                try:
                    _doc_idx, doc = next(doc_iter)
                except StopIteration:
                    break

                doc_id = str(doc.get("id", ""))
                text = str(doc.get("text", ""))
                if not doc_id:
                    continue

                total_docs += 1
                units, next_unit_id = splitter.split_document_streaming(
                    doc_id=doc_id,
                    text=text,
                    unit_id_start=next_unit_id,
                    nlp=split_components["nlp"],
                    max_chars_per_batch=split_components["max_chars_per_batch"],
                )
                total_units += len(units)
                if split_save_output and split_path_obj is not None and units:
                    self._append_jsonl_rows(split_path_obj, units)
                if units:
                    self._append_jsonl_rows(window_paths.split_tmp, units)

                window_doc_ids.append(doc_id)
                split_progress.update(1)
                split_progress.set_postfix(
                    doc=doc_id,
                    doc_units=len(units),
                    total_units=total_units,
                    refresh=True,
                )
        finally:
            split_progress.close()

        return window_doc_ids, next_unit_id, total_docs, total_units

    def _run_split_window_from_saved_split(
        self,
        *,
        doc_iter: Any,
        split_units_iter: Any,
        next_saved_units_pair: tuple[str, list[dict[str, Any]]] | None,
        next_unit_id: int,
        window_paths: StreamingWindowPaths,
        next_doc_idx: int,
        end_doc_idx_exclusive: int,
        total_docs: int,
        total_units: int,
    ) -> tuple[list[str], int, int, int, tuple[str, list[dict[str, Any]]] | None]:
        window_target_docs = max(0, end_doc_idx_exclusive - next_doc_idx)
        window_doc_ids: list[str] = []
        split_progress = tqdm(
            total=window_target_docs,
            desc="Split (streaming, reuse)",
            unit="doc",
            dynamic_ncols=True,
            miniters=1,
        )
        try:
            while len(window_doc_ids) < window_target_docs:
                try:
                    _doc_idx, doc = next(doc_iter)
                except StopIteration:
                    break

                normalized_doc_id = str(doc.get("id", ""))
                if not normalized_doc_id:
                    continue

                units: list[dict[str, Any]] = []
                if next_saved_units_pair is not None and str(next_saved_units_pair[0]) == normalized_doc_id:
                    units = next_saved_units_pair[1]
                    next_saved_units_pair = next(split_units_iter, None)

                normalized_units: list[dict[str, Any]] = []
                max_unit_id = next_unit_id
                for unit in units:
                    if not isinstance(unit, dict):
                        continue
                    normalized_units.append(unit)
                    unit_id = unit.get("unit_id")
                    if isinstance(unit_id, int) and unit_id >= max_unit_id:
                        max_unit_id = unit_id + 1

                total_docs += 1
                total_units += len(normalized_units)
                next_unit_id = max_unit_id
                if normalized_units:
                    self._append_jsonl_rows(window_paths.split_tmp, normalized_units)

                window_doc_ids.append(normalized_doc_id)
                split_progress.update(1)
                split_progress.set_postfix(
                    doc=normalized_doc_id,
                    doc_units=len(normalized_units),
                    total_units=total_units,
                    refresh=True,
                )
        finally:
            split_progress.close()

        return window_doc_ids, next_unit_id, total_docs, total_units, next_saved_units_pair

    def _run_chunking_window(
        self,
        *,
        chunker: Any,
        validated_chunk_cfg: dict[str, Any],
        run_dir: Path,
        window_paths: StreamingWindowPaths,
        window_doc_ids: list[str],
        router: Any | None,
        router_cfg: dict[str, Any],
        split_type: str,
        split_components: dict[str, Any],
        intrinsic_runtime: dict[str, Any] | None,
        total_chunks: int,
    ) -> tuple[int, int]:
        chunk_progress = tqdm(
            total=len(window_doc_ids),
            desc="Chunking (streaming)",
            unit="doc",
            dynamic_ncols=True,
            miniters=1,
        )
        try:
            units_iter = (
                iter(self._iter_split_units_grouped_by_doc(window_paths.split_tmp))
                if window_paths.split_tmp.exists()
                else iter(())
            )
            next_units_pair = next(units_iter, None)
            window_chunk_count = 0

            for doc_id in window_doc_ids:
                if next_units_pair is not None and next_units_pair[0] == doc_id:
                    units = next_units_pair[1]
                    next_units_pair = next(units_iter, None)
                else:
                    units = []

                routed_units = self._run_router_window(
                    units=units,
                    doc_id=doc_id,
                    router=router,
                    router_cfg=router_cfg,
                    split_type=split_type,
                    split_components=split_components,
                )
                grouped = chunker.group_split_units(routed_units)
                if hasattr(chunker, "ensure_semantic_embeddings"):
                    chunker.ensure_semantic_embeddings(
                        split_units=routed_units,
                        grouped_units=grouped,
                        config=validated_chunk_cfg,
                        split_path=None,
                    )
                chunks, _ = chunker.chunk_grouped_units(grouped, validated_chunk_cfg)
                total_chunks += len(chunks)
                window_chunk_count += len(chunks)
                if validated_chunk_cfg.get("save_chunks", False):
                    chunker.save_run_chunks(run_dir=run_dir, grouped_chunks=chunks, config=validated_chunk_cfg)
                if intrinsic_runtime is not None:
                    self._update_streaming_intrinsic_runtime(intrinsic_runtime, doc_id=doc_id, chunks=chunks)
                if chunks:
                    self._append_jsonl_rows(window_paths.chunks_tmp, chunks)

                chunk_progress.update(1)
                chunk_progress.set_postfix(
                    doc=doc_id,
                    doc_chunks=len(chunks),
                    total_chunks=total_chunks,
                    refresh=True,
                )
        finally:
            chunk_progress.close()
            if window_paths.split_tmp.exists():
                window_paths.split_tmp.unlink()

        return window_chunk_count, total_chunks

    def _run_router_window(
        self,
        *,
        units: list[dict[str, Any]],
        doc_id: str,
        router: Any | None,
        router_cfg: dict[str, Any],
        split_type: str,
        split_components: dict[str, Any],
    ) -> list[dict[str, Any]]:
        # Placeholder stage for future window-level routing policies.
        # For now, routing remains document-level and reuses the existing logic.
        return self._route_units_for_streaming_doc(
            units=units,
            doc_id=doc_id,
            router=router,
            router_cfg=router_cfg,
            split_type=split_type,
            split_components=split_components,
        )

    def _route_units_for_streaming_doc(
        self,
        *,
        units: list[dict[str, Any]],
        doc_id: str,
        router: Any | None,
        router_cfg: dict[str, Any],
        split_type: str,
        split_components: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if router is None:
            return units

        routed_output = router.route(
            {
                "split_units": units,
                "documents": [{"doc_id": doc_id, "num_sentences": len(units)}],
                "metadata": {
                    "split_type": split_type,
                    "spacy_model": split_components["model_name"],
                    "streaming": True,
                },
            },
            router_cfg,
        )
        if not isinstance(routed_output, dict) or not isinstance(routed_output.get("split_units"), list):
            raise TypeError("Router output must be a dict containing 'split_units' list.")
        return routed_output["split_units"]

    def _run_indexing_window(
        self,
        *,
        window_paths: StreamingWindowPaths,
        retriever: Any | None,
        embedder: Any | None,
        stream_embedding_cfg: dict[str, Any],
        window_chunk_count: int,
        total_index_items: int,
    ) -> int:
        if retriever is None or embedder is None:
            if window_paths.chunks_tmp.exists():
                window_paths.chunks_tmp.unlink()
            return total_index_items

        index_progress = tqdm(
            total=window_chunk_count,
            desc="Embedding+Index (streaming)",
            unit="chunk",
            dynamic_ncols=True,
            miniters=1,
        )
        try:
            if window_paths.chunks_tmp.exists():
                batch_size = int(stream_embedding_cfg.get("batch_size", 32))
                if batch_size <= 0:
                    batch_size = 32

                batch_chunks: list[dict[str, Any]] = []
                batch_texts: list[str] = []
                last_doc_id = ""
                with window_paths.chunks_tmp.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        chunk = json.loads(line)
                        doc_id = str(chunk.get("doc_id", ""))
                        text = str(chunk.get("text", "")).strip()
                        last_doc_id = doc_id or last_doc_id
                        if text:
                            batch_chunks.append(chunk)
                            batch_texts.append(text)

                        index_progress.update(1)

                        if len(batch_texts) >= batch_size:
                            total_index_items = self._flush_embedding_batch(
                                retriever=retriever,
                                embedder=embedder,
                                texts=batch_texts,
                                chunks=batch_chunks,
                                stream_embedding_cfg=stream_embedding_cfg,
                                total_index_items=total_index_items,
                                index_progress=index_progress,
                                last_doc_id=last_doc_id,
                            )

                if batch_texts:
                    total_index_items = self._flush_embedding_batch(
                        retriever=retriever,
                        embedder=embedder,
                        texts=batch_texts,
                        chunks=batch_chunks,
                        stream_embedding_cfg=stream_embedding_cfg,
                        total_index_items=total_index_items,
                        index_progress=index_progress,
                        last_doc_id=last_doc_id,
                    )
        finally:
            index_progress.close()
            if window_paths.chunks_tmp.exists():
                window_paths.chunks_tmp.unlink()
        return total_index_items

    def _flush_embedding_batch(
        self,
        *,
        retriever: Any,
        embedder: Any,
        texts: list[str],
        chunks: list[dict[str, Any]],
        stream_embedding_cfg: dict[str, Any],
        total_index_items: int,
        index_progress: Any,
        last_doc_id: str,
    ) -> int:
        vectors, emb_metadata = embedder.encode_texts(texts, stream_embedding_cfg)
        items_for_index = [self._serialize_chunk_item_for_index(chunk_item) for chunk_item in chunks]
        retriever.add_embeddings_batch(vectors, items_for_index, emb_metadata)
        total_index_items += len(items_for_index)
        texts.clear()
        chunks.clear()
        index_progress.set_postfix(
            doc=last_doc_id,
            index_items=total_index_items,
            refresh=True,
        )
        return total_index_items

    def _run_dataset_processing(self) -> Any:
        dataset_cfg = dict(self.config["dataset"])
        requested_tasks = (
            self.config.get("evaluation", {}).get("extrinsic_tasks_to_run")
            or [self.config.get("evaluation", {}).get("extrinsic_evaluator", "document_retrieval")]
        )
        if "required_extrinsic_tasks" not in dataset_cfg:
            dataset_cfg["required_extrinsic_tasks"] = requested_tasks
        dataset_name = dataset_cfg["name"]

        print(f"[INFO] Processing dataset: {dataset_name}")
        processor = DatasetFactory.create(dataset_name)
        if self._should_force_rebuild("dataset"):
            print("[INFO] Dataset reuse disabled by execution policy: rebuilding from raw.")
            data = processor.build_from_raw(dataset_cfg)
            return processor.finalize_raw_data(data, dataset_cfg)

        if self._should_allow_reuse("dataset"):
            reusable_output = processor.try_load_reusable_normalized(dataset_cfg)
            if reusable_output is not None:
                print("[INFO] Dataset shortcut activated: reusing normalized dataset.")
                return reusable_output
        else:
            print("[INFO] Dataset reuse disabled by execution policy: skipping normalized shortcut.")

        data = processor.build_from_raw(dataset_cfg)
        return processor.finalize_raw_data(data, dataset_cfg)

    def _find_reusable_index(self, chunk_output: Any) -> Optional[Dict[str, Any]]:
        retrieval_cfg = self.config.get("retrieval", {})
        if not retrieval_cfg.get("enabled", True):
            return None
        if self._should_force_rebuild("retrieval"):
            print("[INFO] Retrieval reuse disabled by execution policy: rebuilding index.")
            return None
        if not self._should_allow_reuse("retrieval"):
            print("[INFO] Retrieval reuse disabled by execution policy: skipping index shortcut.")
            return None

        output_dir = self._resolve_retrieval_output_dir(retrieval_cfg)
        manifest_path = output_dir / "manifest.json"
        metadata_path = output_dir / "metadata.pkl"

        if not manifest_path.exists() or not metadata_path.exists():
            return None

        try:
            with manifest_path.open("r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception as e:
            print(f"[INFO] Existing index manifest is unreadable, rebuilding index: {e}")
            return None

        if not isinstance(manifest, dict):
            print("[INFO] Existing index manifest is invalid, rebuilding index.")
            return None

        expected_num_items = self._count_chunk_items(chunk_output)
        if not self._manifest_matches_current_config(
            manifest=manifest,
            metadata_path=metadata_path,
            expected_num_items=expected_num_items,
        ):
            return None

        index_path = manifest.get("index_path")
        if not isinstance(index_path, str) or not index_path:
            print("[INFO] Existing index manifest is missing 'index_path', rebuilding index.")
            return None

        index_path_obj = Path(index_path)
        if not index_path_obj.exists():
            print(f"[INFO] Existing index file not found at {index_path_obj}, rebuilding index.")
            return None

        print(f"[INFO] Index shortcut activated: reusing existing index at {index_path_obj}")
        return {
            "index_path": str(index_path_obj),
            "metadata_path": str(metadata_path),
            "manifest_path": str(manifest_path),
            "num_vectors": manifest.get("num_vectors"),
            "dimension": manifest.get("dimension"),
            "distance": manifest.get("distance"),
        }

    def _resolve_retrieval_output_dir(self, retrieval_cfg: Dict[str, Any]) -> Path:
        dataset_name = self.config["dataset"]["name"]
        chunking_type = self.config["chunking"]["type"]
        configured_output_dir = retrieval_cfg.get("output_dir")
        if isinstance(configured_output_dir, str) and configured_output_dir.strip():
            return Path(configured_output_dir)

        if self.config_path is not None:
            experiment_id = self.config_path.stem
        else:
            experiment_id = str(self.config.get("experiment_name", "")).strip()

        if not experiment_id:
            experiment_id = str(self.config.get("embedding", {}).get("name", "default")).strip() or "default"

        output_dir = f"data/indexes/{dataset_name}/{chunking_type}/{experiment_id}"
        return Path(output_dir)

    def _manifest_matches_current_config(
        self,
        *,
        manifest: Dict[str, Any],
        metadata_path: Path,
        expected_num_items: int,
    ) -> bool:
        if manifest.get("schema_version") != 1:
            print("[INFO] Existing index manifest schema mismatch, rebuilding index.")
            return False

        retrieval_cfg = self.config.get("retrieval", {})

        expected_pairs = {
            "retriever": retrieval_cfg.get("name", "faiss"),
            "dataset": self.config["dataset"]["name"],
            "chunking": self.config["chunking"]["type"],
            "embedder": self.config["embedding"]["name"],
            "distance": retrieval_cfg.get("distance", "cosine").lower(),
            "normalize": retrieval_cfg.get("normalize", True),
            "num_items": expected_num_items,
        }

        for key, expected_value in expected_pairs.items():
            actual_value = manifest.get(key)
            if actual_value != expected_value:
                print(
                    "[INFO] Existing index manifest mismatch on "
                    f"{key}: expected={expected_value!r}, found={actual_value!r}. Rebuilding index."
                )
                return False

        try:
            import pickle

            with metadata_path.open("rb") as f:
                metadata_blob = pickle.load(f)
        except Exception as e:
            print(f"[INFO] Existing index metadata is unreadable, rebuilding index: {e}")
            return False

        if not isinstance(metadata_blob, dict):
            print("[INFO] Existing index metadata is invalid, rebuilding index.")
            return False

        items = metadata_blob.get("items")
        if isinstance(items, list):
            found_items = len(items)
        else:
            found_items = metadata_blob.get("num_items")
            if not isinstance(found_items, int):
                items_jsonl = metadata_blob.get("items_jsonl_path")
                if isinstance(items_jsonl, str) and items_jsonl:
                    try:
                        with Path(items_jsonl).open("r", encoding="utf-8") as f:
                            found_items = sum(1 for line in f if line.strip())
                    except Exception:
                        found_items = None
                else:
                    found_items = None

        if found_items is None:
            print("[INFO] Existing index metadata is missing item count information, rebuilding index.")
            return False
        if int(found_items) != expected_num_items:
            print(
                "[INFO] Existing index metadata item count mismatch, "
                f"expected={expected_num_items}, found={found_items}. Rebuilding index."
            )
            return False

        return True

    def _execution_section(self) -> Dict[str, Any]:
        execution = self.config.get("execution", {})
        return execution if isinstance(execution, dict) else {}

    def _stage_execution_config(self, stage_name: str) -> Dict[str, Any]:
        execution = self._execution_section()
        stages = execution.get("stages", {})
        if not isinstance(stages, dict):
            return {}
        stage_cfg = stages.get(stage_name, {})
        return stage_cfg if isinstance(stage_cfg, dict) else {}

    def _should_force_rebuild(self, stage_name: str) -> bool:
        stage_cfg = self._stage_execution_config(stage_name)
        if "force_rebuild" in stage_cfg:
            return bool(stage_cfg.get("force_rebuild"))

        execution = self._execution_section()
        return bool(execution.get("force_rebuild", False))

    def _should_allow_reuse(self, stage_name: str) -> bool:
        if self._should_force_rebuild(stage_name):
            return False

        stage_cfg = self._stage_execution_config(stage_name)
        if "allow_reuse" in stage_cfg:
            return bool(stage_cfg.get("allow_reuse"))

        execution = self._execution_section()
        return bool(execution.get("allow_reuse", True))

    @staticmethod
    def _count_chunk_items(chunk_output: Any) -> int:
        if isinstance(chunk_output, dict):
            chunks = chunk_output.get("chunks", [])
            if isinstance(chunks, list):
                if len(chunks) == 0:
                    metadata = chunk_output.get("metadata", {})
                    if isinstance(metadata, dict) and isinstance(metadata.get("total_chunks"), int):
                        return int(metadata["total_chunks"])
                return len(chunks)
            metadata = chunk_output.get("metadata", {})
            if isinstance(metadata, dict) and isinstance(metadata.get("total_chunks"), int):
                return int(metadata["total_chunks"])
            raise TypeError("chunk_output['chunks'] must be a list, or metadata.total_chunks must be present.")

        if isinstance(chunk_output, list):
            return len(chunk_output)

        raise TypeError("chunk_output must be either a dict with 'chunks' or a list.")

    def _iter_split_units_grouped_by_doc(self, split_path: Path) -> Iterable[tuple[str, list[dict[str, Any]]]]:
        current_doc_id: str | None = None
        current_units: list[dict[str, Any]] = []

        with split_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                unit = json.loads(line)
                doc_id = str(unit.get("doc_id", ""))
                if not doc_id:
                    continue

                if current_doc_id is None:
                    current_doc_id = doc_id

                if doc_id != current_doc_id:
                    yield current_doc_id, current_units
                    current_doc_id = doc_id
                    current_units = []

                current_units.append(unit)

        if current_doc_id is not None and current_units:
            yield current_doc_id, current_units

    def _iter_documents_from_normalized(self, dataset_output: Any) -> Iterable[dict[str, Any]]:
        if not isinstance(dataset_output, dict):
            raise TypeError("dataset_output must be a dict in streaming mode.")
        metadata = dataset_output.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("dataset_output['metadata'] must be a dict in streaming mode.")
        normalized_path = metadata.get("normalized_path")
        normalized_files = metadata.get("normalized_files", {})
        if not isinstance(normalized_path, str) or not normalized_path:
            raise ValueError("dataset_output metadata is missing 'normalized_path'.")
        if not isinstance(normalized_files, dict):
            raise ValueError("dataset_output metadata is missing valid 'normalized_files'.")
        documents_rel = normalized_files.get("documents")
        if not isinstance(documents_rel, str) or not documents_rel:
            raise ValueError("dataset_output metadata is missing normalized documents filename.")

        documents_path = Path(normalized_path) / documents_rel
        with documents_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)

    @staticmethod
    def _append_jsonl_rows(path: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    @staticmethod
    def _serialize_chunk_item_for_index(chunk: dict[str, Any]) -> dict[str, Any]:
        metadata = chunk.get("metadata", {}) if isinstance(chunk.get("metadata"), dict) else {}
        for field in ("sentences", "start_sentence_idx", "end_sentence_idx", "position"):
            if field in chunk:
                metadata[field] = chunk[field]
        return {
            "item_id": str(chunk.get("chunk_id", "")),
            "doc_id": str(chunk.get("doc_id", "")),
            "text": str(chunk.get("text", "")),
            "metadata": metadata,
        }

    def _init_streaming_intrinsic_runtime(self, intrinsic_cfg: dict[str, Any]) -> dict[str, Any]:
        model_cfg = intrinsic_cfg.get("intrinsic_model", {})
        metrics_cfg = intrinsic_cfg.get("intrinsic_metrics", {})
        from src.evaluation.intrinsic.models.perplexity_scorer import PerplexityScorer

        return {
            "scorer": PerplexityScorer.from_config(model_cfg),
            "bc_enabled": bool(metrics_cfg.get("boundary_clarity", True)),
            "cs_enabled": bool(metrics_cfg.get("chunk_stickiness", True)),
            "bc_sum": 0.0,
            "bc_count": 0,
            "csc_sum": 0.0,
            "csc_count": 0,
            "csi_sum": 0.0,
            "csi_count": 0,
            "num_documents": 0,
            "num_chunks": 0,
        }

    def _update_streaming_intrinsic_runtime(
        self,
        runtime: dict[str, Any],
        *,
        doc_id: str,
        chunks: list[dict[str, Any]],
    ) -> None:
        from src.evaluation.intrinsic.metrics.boundary_clarity import compute_boundary_clarity
        from src.evaluation.intrinsic.metrics.chunk_stickiness import compute_chunk_stickiness

        doc = {"doc_id": doc_id, "chunks": chunks}
        runtime["num_documents"] += 1
        runtime["num_chunks"] += len(chunks)

        if runtime["bc_enabled"]:
            bc_score = compute_boundary_clarity(doc=doc, scorer=runtime["scorer"], config=self.config.get("evaluation", {}))
            runtime["bc_sum"] += float(bc_score)
            runtime["bc_count"] += 1

        if runtime["cs_enabled"]:
            cs_result = compute_chunk_stickiness(doc=doc, scorer=runtime["scorer"], config=self.config.get("evaluation", {}))
            runtime["csc_sum"] += float(cs_result["complete"])
            runtime["csc_count"] += 1
            runtime["csi_sum"] += float(cs_result["incomplete"])
            runtime["csi_count"] += 1

    def _finalize_streaming_intrinsic_runtime(
        self,
        runtime: dict[str, Any],
        intrinsic_cfg: dict[str, Any],
    ) -> dict[str, Any]:
        aggregated_metrics: dict[str, float] = {}
        if runtime["bc_count"] > 0:
            aggregated_metrics["boundary_clarity"] = runtime["bc_sum"] / float(runtime["bc_count"])
        if runtime["csc_count"] > 0:
            aggregated_metrics["chunk_stickiness_complete"] = runtime["csc_sum"] / float(runtime["csc_count"])
        if runtime["csi_count"] > 0:
            aggregated_metrics["chunk_stickiness_incomplete"] = runtime["csi_sum"] / float(runtime["csi_count"])

        model_cfg = intrinsic_cfg.get("intrinsic_model", {})
        results = {
            "evaluator_name": "default",
            "metrics": aggregated_metrics,
            "per_document_metrics": [],
            "metadata": {
                "num_documents": runtime["num_documents"],
                "num_chunks": runtime["num_chunks"],
                "scorer_backend": runtime["scorer"].backend_name,
                "intrinsic_model": {
                    "edge_threshold": float(model_cfg.get("edge_threshold", 0.8)),
                    "sequential_delta": int(model_cfg.get("sequential_delta", 0)),
                },
                "streaming": True,
                "stores_per_document_metrics": False,
            },
        }

        evaluator_name = intrinsic_cfg.get("intrinsic_evaluator", "default")
        evaluator = IntrinsicEvaluatorFactory.create(evaluator_name)
        if hasattr(evaluator, "_save_results"):
            save_cfg = dict(intrinsic_cfg)
            save_cfg["_run_context"] = {
                "dataset": self.config.get("dataset", {}),
                "chunking": self.config.get("chunking", {}),
                "router": self.config.get("router", {}),
                "experiment_name": self.config.get("experiment_name"),
                "experiment": self.config.get("experiment", {}),
                "config_path": str(self.config_path) if self.config_path is not None else None,
            }
            evaluator._save_results(results, save_cfg)

        return results

    def _streaming_docs_per_run(self) -> int:
        execution = self._execution_section()
        raw = execution.get("streaming_docs_per_run", execution.get("window_docs_per_run"))
        if raw is None:
            return 0
        try:
            value = int(raw)
        except Exception:
            raise ValueError("'execution.streaming_docs_per_run' must be an integer > 0.")
        if value <= 0:
            raise ValueError("'execution.streaming_docs_per_run' must be > 0 when provided.")
        return value

    def _streaming_run_until_complete(self) -> bool:
        execution = self._execution_section()
        return bool(execution.get("streaming_run_until_complete", True))

    def _resolve_stream_state_path(self, retrieval_cfg: dict[str, Any]) -> Path:
        output_dir = self._resolve_retrieval_output_dir(retrieval_cfg)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / "stream_state.json"

    def _load_stream_state(self, state_path: Path) -> dict[str, Any]:
        if not state_path.exists():
            return {}
        try:
            data = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return data if isinstance(data, dict) else {}

    def _save_stream_state(self, state_path: Path, payload: dict[str, Any]) -> None:
        state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _resolve_split_state_path(self, split_path: Path | None) -> Path | None:
        if split_path is None:
            return None
        return split_path.parent / "split_state.json"

    def _load_split_state(self, state_path: Path | None) -> dict[str, Any]:
        if state_path is None or not state_path.exists():
            return {}
        try:
            data = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return data if isinstance(data, dict) else {}

    def _save_split_state(self, state_path: Path | None, payload: dict[str, Any]) -> None:
        if state_path is None:
            return
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _serialize_intrinsic_runtime_for_state(self, runtime: dict[str, Any] | None) -> dict[str, Any] | None:
        if not isinstance(runtime, dict):
            return None
        return {
            "bc_sum": float(runtime.get("bc_sum", 0.0)),
            "bc_count": int(runtime.get("bc_count", 0)),
            "csc_sum": float(runtime.get("csc_sum", 0.0)),
            "csc_count": int(runtime.get("csc_count", 0)),
            "csi_sum": float(runtime.get("csi_sum", 0.0)),
            "csi_count": int(runtime.get("csi_count", 0)),
            "num_documents": int(runtime.get("num_documents", 0)),
            "num_chunks": int(runtime.get("num_chunks", 0)),
        }

    def _run_extrinsic_evaluation(self, chunk_output: Any, index_output: Any) -> None:
        eval_cfg = self.config.get("evaluation", {})
        extrinsic_enabled = eval_cfg.get("extrinsic", True)

        if not extrinsic_enabled:
            print("[INFO] Extrinsic evaluation disabled.")
            return

        script_path = eval_cfg.get("extrinsic_script", "scripts/run_extrinsic_eval.py")

        print(f"[INFO] Running extrinsic evaluation script: {script_path}")

        module_name = script_path.replace("/", ".").replace("\\", ".")
        if module_name.endswith(".py"):
            module_name = module_name[:-3]

        cmd = [
            "python",
            "-m",
            module_name,
        ]

        if self.config_path is not None:
            cmd.extend(["--config", str(self.config_path)])

        if index_output is not None and isinstance(index_output, dict):
            index_path = index_output.get("index_path")
            metadata_path = index_output.get("metadata_path")
            manifest_path = index_output.get("manifest_path")

            if index_path:
                cmd.extend(["--index-path", str(index_path)])
            if metadata_path:
                cmd.extend(["--index-metadata-path", str(metadata_path)])
            if manifest_path:
                cmd.extend(["--index-manifest-path", str(manifest_path)])

        subprocess.run(cmd, check=True)
