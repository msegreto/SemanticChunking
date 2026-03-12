from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.evaluation.intrinsic.base import BaseIntrinsicEvaluator
from src.evaluation.intrinsic.metrics.boundary_clarity import compute_boundary_clarity
from src.evaluation.intrinsic.metrics.chunk_stickiness import compute_chunk_stickiness
from src.evaluation.intrinsic.metrics.stats import build_global_statistics
from src.evaluation.intrinsic.metrics.utils import count_total_chunks, normalize_chunk_output
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
            doc_result: Dict[str, Any] = {
                "doc_id": doc["doc_id"],
                "num_chunks": len(doc["chunks"]),
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
            },
        }

        self._save_results(results, config)
        return results

    def _save_results(self, results: Dict[str, Any], config: Dict[str, Any]) -> None:
        save_cfg = config.get("save", {})
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

        yaml_name = save_cfg.get("yaml_name")
        if yaml_name:
            yaml_stem = Path(yaml_name).stem
            main_name = f"intrinsic_{yaml_stem}.json"
            global_name = f"intrinsic_{yaml_stem}_global.json"
        else:
            main_name = "intrinsic_results.json"
            global_name = "intrinsic_results_global.json"

        return output_dir / main_name, output_dir / global_name