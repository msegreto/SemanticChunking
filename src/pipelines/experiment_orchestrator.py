from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

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

        split_output = self._run_split(dataset_output)
        self.artifacts.split_output = split_output

        routed_output = self._run_router_if_needed(split_output)
        self.artifacts.routed_output = routed_output

        chunk_output = self._run_chunking(routed_output)
        self.artifacts.chunk_output = chunk_output

        intrinsic_output = self._run_intrinsic_evaluation(chunk_output)
        self.artifacts.intrinsic_output = intrinsic_output

        embedding_output = self._run_embedding(chunk_output)
        self.artifacts.embedding_output = embedding_output

        index_output = self._run_indexing(embedding_output)
        self.artifacts.index_output = index_output

        self._run_extrinsic_evaluation(
            chunk_output=chunk_output,
            embedding_output=embedding_output,
            index_output=index_output,
        )

        print("[INFO] Pipeline completed successfully.")

    def _run_dataset_processing(self) -> Any:
        dataset_cfg = self.config["dataset"]
        dataset_name = dataset_cfg["name"]

        print(f"[INFO] Processing dataset: {dataset_name}")
        processor = DatasetFactory.create(dataset_name)
        return processor.process(dataset_cfg)

    def _run_split(self, dataset_output: Any) -> Any:
        split_cfg = self.config["split"]
        split_type = split_cfg["type"]

        print(f"[INFO] Running split: {split_type}")
        splitter = SplitterFactory.create(split_type)
        return splitter.split(dataset_output, split_cfg)

    def _run_router_if_needed(self, split_output: Any) -> Any:
        split_type = self.config["split"]["type"]
        router_cfg = self.config.get("router", {})
        router_enabled = router_cfg.get("enabled", False)

        if split_type == "proposition":
            print("[INFO] Proposition split selected: skipping router.")
            return split_output

        if not router_enabled:
            print("[INFO] Router disabled: skipping router.")
            return split_output

        router_name = router_cfg["name"]
        print(f"[INFO] Running router: {router_name}")
        router = RouterFactory.create(router_name)
        return router.route(split_output, router_cfg)

    def _run_chunking(self, routed_output: Any) -> Any:
        chunking_cfg = self.config["chunking"]
        chunking_type = chunking_cfg["type"]

        print(f"[INFO] Running chunking: {chunking_type}")
        chunker = ChunkerFactory.create(chunking_type)
        return chunker.chunk(routed_output, chunking_cfg)

    def _run_intrinsic_evaluation(self, chunk_output: Any) -> Any:
        eval_cfg = self.config.get("evaluation", {})
        intrinsic_enabled = eval_cfg.get("intrinsic", True)

        if not intrinsic_enabled:
            print("[INFO] Intrinsic evaluation disabled.")
            return None

        intrinsic_name = eval_cfg.get("intrinsic_evaluator", "default")
        print(f"[INFO] Running intrinsic evaluation: {intrinsic_name}")
        evaluator = IntrinsicEvaluatorFactory.create(intrinsic_name)
        return evaluator.evaluate(chunk_output, eval_cfg)

    def _run_embedding(self, chunk_output: Any) -> Any:
        embedding_cfg = self.config["embedding"]
        embedder_name = embedding_cfg["name"]

        print(f"[INFO] Running embedder: {embedder_name}")
        embedder = EmbedderFactory.create(embedder_name)
        return embedder.embed(chunk_output, embedding_cfg)

    def _run_indexing(self, embedding_output: Any) -> Any:
        retrieval_cfg = self.config.get("retrieval", {})
        index_enabled = retrieval_cfg.get("enabled", True)

        if not index_enabled:
            print("[INFO] Indexing disabled.")
            return None

        retriever_name = retrieval_cfg.get("name", "faiss")
        print(f"[INFO] Building retrieval index: {retriever_name}")

        retriever = RetrieverFactory.create(retriever_name)
        return retriever.build_index(embedding_output, retrieval_cfg, self.config)

    def _run_extrinsic_evaluation(self, chunk_output: Any, embedding_output: Any, index_output: Any) -> None:
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