from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import pyterrier as pt
except ImportError:  # pragma: no cover
    pt = None

from src.chunking import ChunkTransformer
from src.config.loader import load_experiment_config
from src.datasets.service import process_dataset, try_load_normalized
from src.evaluation.extrinsic import ExtrinsicEvaluationTransformer
from src.evaluation.intrinsic import IntrinsicEvaluationTransformer
from src.retrieval import RetrievalTransformer
from src.splitting import SplitTransformer


_PtTransformerBase = pt.Transformer if pt is not None else object


class _ArtifactTapTransformer(_PtTransformerBase):
    def __init__(self, callback: Any) -> None:
        self._callback = callback

    def transform(self, inp: Any) -> Any:
        self._callback(inp)
        return inp


class _DatasetAttachTransformer(_PtTransformerBase):
    def __init__(self, dataset: Any) -> None:
        self._dataset = dataset

    def transform(self, inp: Any) -> dict[str, Any]:
        return {
            "chunk_output": inp,
            "dataset": self._dataset,
        }


@dataclass
class ExperimentArtifacts:
    dataset_output: Any = None
    split_output: Any = None
    routed_output: Any = None
    chunk_output: Any = None
    intrinsic_output: Any = None
    index_output: Any = None
    extrinsic_output: Any = None


class ExperimentOrchestrator:
    def __init__(self, config: dict[str, Any], config_path: Path | None = None) -> None:
        self.config = config
        self.config_path = config_path
        self.artifacts = ExperimentArtifacts()

    @classmethod
    def from_yaml(cls, config_path: Path) -> "ExperimentOrchestrator":
        config = load_experiment_config(config_path)
        return cls(config=config, config_path=config_path)

    def run(self) -> None:
        print("[INFO] Starting experiment pipeline...")

        dataset_output = self._run_dataset_processing()
        self.artifacts.dataset_output = dataset_output

        dataset_name = str(self.config.get("dataset", {}).get("name", ""))
        run_name = self._build_run_name()
        pt_dataset = dataset_output.get("pt_dataset") if isinstance(dataset_output, dict) else None
        split_input = pt_dataset if pt_dataset is not None else dataset_output
        if pt is None:
            raise ImportError("PyTerrier is required to run the composed experiment pipeline.")

        split_transformer = SplitTransformer.from_config(
            split_config=self.config["split"],
            dataset_name=dataset_name,
            run_name=run_name,
            allow_reuse=self._should_allow_reuse("split"),
            force_rebuild=self._should_force_rebuild("split"),
        )
        chunk_transformer = ChunkTransformer.from_config(
            chunking_config=self._build_chunking_config(),
            allow_reuse=self._should_allow_reuse("chunking"),
            force_rebuild=self._should_force_rebuild("chunking"),
        )
        split_output, chunk_output, index_output = self._run_core_pipeline(
            split_transformer=split_transformer,
            chunk_transformer=chunk_transformer,
            split_input=split_input,
            pt_dataset=pt_dataset,
            dataset_name=dataset_name,
            run_name=run_name,
        )
        self.artifacts.split_output = split_output
        self.artifacts.routed_output = split_output
        self.artifacts.chunk_output = chunk_output

        intrinsic_output = self._run_intrinsic(chunk_output)
        self.artifacts.intrinsic_output = intrinsic_output
        self.artifacts.index_output = index_output

        extrinsic_output = self._run_extrinsic(index_output)
        self.artifacts.extrinsic_output = extrinsic_output

        print("[INFO] Pipeline completed successfully.")

    def _run_core_pipeline(
        self,
        *,
        split_transformer: SplitTransformer,
        chunk_transformer: ChunkTransformer,
        split_input: Any,
        pt_dataset: Any,
        dataset_name: str,
        run_name: str,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
        router_cfg = self.config.get("router", {})
        router_enabled = isinstance(router_cfg, dict) and bool(router_cfg.get("enabled", False))
        if router_enabled:
            raise NotImplementedError("Router is not supported in the composed PyTerrier pipeline yet.")
        retrieval_cfg = self._build_retrieval_config()
        retrieval_enabled = bool(retrieval_cfg.get("enabled", True))

        split_box: dict[str, Any] = {}
        chunk_box: dict[str, Any] = {}
        pipeline: Any = (
            split_transformer
            >> _ArtifactTapTransformer(lambda value: split_box.setdefault("value", value))
            >> chunk_transformer
            >> _ArtifactTapTransformer(lambda value: chunk_box.setdefault("value", value))
        )
        if retrieval_enabled:
            print("[INFO] Running composed PyTerrier pipeline: split >> chunking >> retrieval.")
            retrieval_transformer = self._build_retrieval_transformer(
                dataset_name=dataset_name,
                run_name=run_name,
            )
            pipeline = (
                pipeline
                >> _DatasetAttachTransformer(pt_dataset)
                >> retrieval_transformer
            )
        else:
            print("[INFO] Running composed PyTerrier pipeline: split >> chunking.")
        index_output = pipeline.transform(split_input)
        split_output = split_box.get("value")
        chunk_output = chunk_box.get("value")
        if not isinstance(split_output, dict):
            raise TypeError("Composed pipeline did not produce a valid split_output dict.")
        if not isinstance(chunk_output, dict):
            raise TypeError("Composed pipeline did not produce a valid chunk_output dict.")
        if not retrieval_enabled:
            index_output = None
        return split_output, chunk_output, index_output

    def _run_intrinsic(self, chunk_output: dict[str, Any]) -> dict[str, Any] | None:
        evaluation_cfg = self.config.get("evaluation", {})
        if not bool(evaluation_cfg.get("intrinsic", True)):
            print("[INFO] Intrinsic evaluation disabled.")
            return None

        print("[INFO] Running intrinsic evaluation transformer.")
        transformer = IntrinsicEvaluationTransformer(
            evaluation_config=evaluation_cfg,
            run_context=self._build_run_context(),
        )
        return transformer.transform(chunk_output)

    def _build_retrieval_transformer(
        self,
        *,
        dataset_name: str,
        run_name: str,
    ) -> RetrievalTransformer:
        return RetrievalTransformer.from_config(
            retrieval_config=self._build_retrieval_config(),
            dataset_name=dataset_name,
            chunking_name=str(self.config.get("chunking", {}).get("type", "")),
            run_name=run_name,
            allow_reuse=self._should_allow_reuse("retrieval"),
            force_rebuild=self._should_force_rebuild("retrieval"),
        )

    def _run_extrinsic(self, index_output: dict[str, Any] | None) -> dict[str, Any] | None:
        if index_output is None:
            return None

        evaluation_cfg = self.config.get("evaluation", {})
        if not bool(evaluation_cfg.get("extrinsic", True)):
            print("[INFO] Extrinsic evaluation disabled.")
            return None

        print("[INFO] Running extrinsic evaluation transformer.")
        transformer = ExtrinsicEvaluationTransformer(
            config=self.config,
            config_path=self.config_path,
        )
        return transformer.transform(index_output)

    def _run_dataset_processing(self) -> dict[str, Any]:
        dataset_cfg = dict(self.config["dataset"])
        requested_tasks = (
            self.config.get("evaluation", {}).get("extrinsic_tasks_to_run")
            or [self.config.get("evaluation", {}).get("extrinsic_evaluator", "document_retrieval")]
        )
        if "required_extrinsic_tasks" not in dataset_cfg:
            dataset_cfg["required_extrinsic_tasks"] = requested_tasks
        dataset_name = dataset_cfg["name"]

        print(f"[INFO] Processing dataset: {dataset_name}")
        if self._should_force_rebuild("dataset"):
            print("[INFO] Dataset reuse disabled by execution policy: rebuilding normalized dataset.")
            return process_dataset(dataset_cfg, force_rebuild=True)

        if self._should_allow_reuse("dataset"):
            reusable_output = try_load_normalized(dataset_cfg)
            if reusable_output is not None:
                print("[INFO] Dataset shortcut activated: reusing normalized dataset.")
                return reusable_output
        else:
            print("[INFO] Dataset reuse disabled by execution policy: skipping normalized shortcut.")

        return process_dataset(dataset_cfg, force_rebuild=False)

    def _build_chunking_config(self) -> dict[str, Any]:
        chunking_cfg = dict(self.config["chunking"])
        embedding_cfg = self.config.get("embedding", {})
        if isinstance(embedding_cfg.get("name"), str) and embedding_cfg["name"].strip():
            chunking_cfg.setdefault("embedding_model", embedding_cfg["name"].strip())
        if isinstance(embedding_cfg.get("show_progress_bar"), bool):
            chunking_cfg.setdefault("show_embedding_progress", embedding_cfg["show_progress_bar"])
        chunking_cfg.setdefault("yaml_name", self._build_run_name(with_suffix=True))
        return chunking_cfg

    def _build_run_context(self) -> dict[str, Any]:
        return {
            "dataset": self.config.get("dataset", {}),
            "chunking": self.config.get("chunking", {}),
            "router": {},
            "experiment_name": self.config.get("experiment_name"),
            "experiment": self.config.get("experiment", {}),
            "config_path": str(self.config_path) if self.config_path is not None else None,
        }

    def _build_run_name(self, *, with_suffix: bool = False) -> str:
        if self.config_path is not None:
            return self.config_path.name if with_suffix else self.config_path.stem
        raw = str(self.config.get("experiment_name", "default_run"))
        return raw if with_suffix else Path(raw).stem

    def _build_retrieval_config(self) -> dict[str, Any]:
        retrieval_cfg = dict(self.config.get("retrieval", {}))
        embedding_cfg = self.config.get("embedding", {})
        if isinstance(embedding_cfg.get("name"), str) and embedding_cfg["name"].strip():
            retrieval_cfg.setdefault("embedding_model", embedding_cfg["name"].strip())
        if isinstance(embedding_cfg.get("batch_size"), int) and embedding_cfg["batch_size"] > 0:
            retrieval_cfg.setdefault("embedding_batch_size", embedding_cfg["batch_size"])
        if isinstance(embedding_cfg.get("show_progress_bar"), bool):
            retrieval_cfg.setdefault("show_progress_bar", embedding_cfg["show_progress_bar"])
        if isinstance(embedding_cfg.get("normalize_embeddings"), bool):
            retrieval_cfg.setdefault("normalize", embedding_cfg["normalize_embeddings"])
        return retrieval_cfg

    def _execution_section(self) -> dict[str, Any]:
        execution = self.config.get("execution", {})
        return execution if isinstance(execution, dict) else {}

    def _stage_execution_config(self, stage_name: str) -> dict[str, Any]:
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
