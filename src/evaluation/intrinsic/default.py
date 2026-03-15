from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.evaluation.intrinsic.base import BaseIntrinsicEvaluator
from src.evaluation.intrinsic.metrics.boundary_clarity import compute_boundary_clarity
from src.evaluation.intrinsic.metrics.chunk_stickiness import compute_chunk_stickiness
from src.evaluation.intrinsic.metrics.stats import build_global_statistics
from src.evaluation.intrinsic.metrics.utils import count_total_chunks, count_words, normalize_chunk_output
from src.evaluation.intrinsic.models.perplexity_scorer import PerplexityScorer


class DefaultIntrinsicEvaluator(BaseIntrinsicEvaluator):
    def evaluate(self, chunk_output: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        docs = normalize_chunk_output(chunk_output)

        model_cfg = config.get("intrinsic_model", {})
        scorer = PerplexityScorer.from_config(model_cfg)

        metrics_cfg = config.get("intrinsic_metrics", {})
        bc_enabled = metrics_cfg.get("boundary_clarity", True)
        cs_enabled = metrics_cfg.get("chunk_stickiness", True)

        per_document_metrics: List[Dict[str, Any]] = []

        bc_values: List[float] = []
        csc_values: List[float] = []
        csi_values: List[float] = []

        for doc in docs:
            chunks = doc["chunks"]
            chunk_texts = [chunk.get("text", "") for chunk in chunks]
            total_chars = sum(len(text) for text in chunk_texts)
            total_words = sum(count_words(text) for text in chunk_texts)
            num_chunks = len(chunks)

            doc_result: Dict[str, Any] = {
                "doc_id": doc["doc_id"],
                "num_chunks": num_chunks,
                "avg_chunk_chars": (float(total_chars) / num_chunks) if num_chunks > 0 else 0.0,
                "avg_chunk_words": (float(total_words) / num_chunks) if num_chunks > 0 else 0.0,
            }

            if bc_enabled:
                bc_score = compute_boundary_clarity(
                    doc=doc,
                    scorer=scorer,
                    config=config,
                )
                doc_result["boundary_clarity"] = bc_score
                bc_values.append(bc_score)

            if cs_enabled:
                cs_result = compute_chunk_stickiness(
                    doc=doc,
                    scorer=scorer,
                    config=config,
                )
                doc_result["chunk_stickiness_complete"] = cs_result["complete"]
                doc_result["chunk_stickiness_incomplete"] = cs_result["incomplete"]
                doc_result["chunk_stickiness_complete_edges"] = cs_result["complete_edges"]
                doc_result["chunk_stickiness_incomplete_edges"] = cs_result["incomplete_edges"]
                csc_values.append(cs_result["complete"])
                csi_values.append(cs_result["incomplete"])

            per_document_metrics.append(doc_result)

        aggregated_metrics: Dict[str, float] = {}

        if bc_values:
            aggregated_metrics["boundary_clarity"] = sum(bc_values) / len(bc_values)

        if csc_values:
            aggregated_metrics["chunk_stickiness_complete"] = sum(csc_values) / len(csc_values)

        if csi_values:
            aggregated_metrics["chunk_stickiness_incomplete"] = sum(csi_values) / len(csi_values)

        results = {
            "evaluator_name": "default",
            "metrics": aggregated_metrics,
            "per_document_metrics": per_document_metrics,
            "metadata": {
                "num_documents": len(docs),
                "num_chunks": count_total_chunks(docs),
                "scorer_backend": scorer.backend_name,
                "intrinsic_model": {
                    "edge_threshold": float(model_cfg.get("edge_threshold", 0.8)),
                    "sequential_delta": int(model_cfg.get("sequential_delta", 0)),
                },
            },
        }

        self._save_results(results, config)
        return results

    def _save_results(self, results: Dict[str, Any], config: Dict[str, Any]) -> None:
        save_cfg = dict(config.get("save", {}))
        save_cfg["_run_context"] = config.get("_run_context", {})
        save_enabled = save_cfg.get("enabled", True)

        if not save_enabled:
            return

        output_dir = Path("results") / "intrinsic"
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path, global_output_path = self._build_output_paths(save_cfg)

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        global_stats = build_global_statistics(results)
        with global_output_path.open("w", encoding="utf-8") as f:
            json.dump(global_stats, f, indent=2, ensure_ascii=False)

    def _build_output_paths(self, save_cfg: Dict[str, Any]) -> Tuple[Path, Path]:
        output_dir = Path("results") / "intrinsic"
        filename_stem = self._build_filename_stem(save_cfg)
        main_name = f"{filename_stem}.json"
        global_name = f"{filename_stem}_global.json"

        return output_dir / main_name, output_dir / global_name

    def _build_filename_stem(self, save_cfg: Dict[str, Any]) -> str:
        run_context = save_cfg.get("_run_context", {})
        if isinstance(run_context, dict):
            dataset_cfg = run_context.get("dataset", {})
            chunking_cfg = run_context.get("chunking", {})
            router_cfg = run_context.get("router", {})
            experiment_cfg = run_context.get("experiment", {})
        else:
            dataset_cfg = {}
            chunking_cfg = {}
            router_cfg = {}
            experiment_cfg = {}

        dataset_name = self._slugify(dataset_cfg.get("name", "unknown-dataset"))
        chunking_name = self._slugify(chunking_cfg.get("type", "unknown-chunking"))

        routing_name = "no-routing"
        if isinstance(router_cfg, dict) and router_cfg.get("enabled", False):
            routing_name = self._slugify(router_cfg.get("name", "default-router"))

        experiment_name = self._slugify(
            save_cfg.get("experiment_name")
            or run_context.get("experiment_name")
            or experiment_cfg.get("name")
            or "intrinsic-eval"
        )

        config_path = run_context.get("config_path")
        if config_path:
            config_name = self._slugify(Path(str(config_path)).stem)
        else:
            config_name = self._slugify(save_cfg.get("yaml_name") or "default-run")

        return f"{dataset_name}_{chunking_name}_{routing_name}_{experiment_name}_{config_name}"

    def _slugify(self, value: Any) -> str:
        text = str(value).replace("/", "-").strip()
        if not text:
            return "item"
        safe_chars = {"-", "_", "."}
        return "".join(ch if ch.isalnum() or ch in safe_chars else "-" for ch in text).strip("-_.") or "item"
