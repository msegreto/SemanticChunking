from __future__ import annotations

import json
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

        index_output = self._find_reusable_index(chunk_output)
        if index_output is not None:
            embedding_output = None
        else:
            embedding_output = self._run_embedding(chunk_output)
            self.artifacts.embedding_output = embedding_output
            index_output = self._run_indexing(embedding_output)

        self.artifacts.embedding_output = embedding_output
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
        chunking_cfg = dict(self.config["chunking"])
        if self.config_path is not None:
            chunking_cfg.setdefault("yaml_name", self.config_path.name)
        else:
            chunking_cfg.setdefault("yaml_name", self.config.get("experiment_name"))
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

    def _find_reusable_index(self, chunk_output: Any) -> Optional[Dict[str, Any]]:
        retrieval_cfg = self.config.get("retrieval", {})
        if not retrieval_cfg.get("enabled", True):
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
        embedder_name = self.config["embedding"]["name"]
        output_dir = retrieval_cfg.get(
            "output_dir",
            f"data/indexes/{dataset_name}/{chunking_type}/{embedder_name}",
        )
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
        if not isinstance(items, list):
            print("[INFO] Existing index metadata is missing a valid items list, rebuilding index.")
            return False

        if len(items) != expected_num_items:
            print(
                "[INFO] Existing index metadata item count mismatch, "
                f"expected={expected_num_items}, found={len(items)}. Rebuilding index."
            )
            return False

        return True

    @staticmethod
    def _count_chunk_items(chunk_output: Any) -> int:
        if isinstance(chunk_output, dict):
            chunks = chunk_output.get("chunks", [])
            if not isinstance(chunks, list):
                raise TypeError("chunk_output['chunks'] must be a list.")
            return len(chunks)

        if isinstance(chunk_output, list):
            return len(chunk_output)

        raise TypeError("chunk_output must be either a dict with 'chunks' or a list.")

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
