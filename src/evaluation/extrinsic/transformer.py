from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

try:
    import pyterrier as pt
except ImportError:  # pragma: no cover
    pt = None

from src.evaluation.extrinsic.answer_generation import AnswerGenerationEvaluator
from src.evaluation.extrinsic.common import build_base_row, build_run_context
from src.evaluation.extrinsic.document_retrieval import DocumentRetrievalEvaluator
from src.evaluation.extrinsic.evidence_retrieval import EvidenceRetrievalEvaluator
from src.evaluation.extrinsic.io import (
    canonicalize_task_name,
    check_document_retrieval_prerequisites,
    get_extrinsic_task_cfg,
    resolve_answers_path,
    resolve_evidence_path,
)


_PtTransformerBase = pt.Transformer if pt is not None else object


class ExtrinsicEvaluationTransformer(_PtTransformerBase):
    _TASKS = {
        "document_retrieval": DocumentRetrievalEvaluator,
        "doc_retrieval": DocumentRetrievalEvaluator,
        "document": DocumentRetrievalEvaluator,
        "evidence_retrieval": EvidenceRetrievalEvaluator,
        "evidence": EvidenceRetrievalEvaluator,
        "answer_generation": AnswerGenerationEvaluator,
        "generation": AnswerGenerationEvaluator,
        "qa_generation": AnswerGenerationEvaluator,
    }

    def __init__(
        self,
        *,
        config: dict[str, Any],
        config_path: Path | None = None,
    ) -> None:
        self.config = config
        self.config_path = config_path

    def transform(self, inp: Any) -> dict[str, Any]:
        return self.build(inp)

    def build(self, retrieval_output: dict[str, Any]) -> dict[str, Any]:
        requested_tasks = self._resolve_requested_tasks(self.config)
        rows: list[dict[str, Any]] = []

        for task_name in requested_tasks:
            ks = self._resolve_ks_for_task(self.config, task_name)
            supported, support_reason = self._task_support_status(self.config, task_name)
            if not supported:
                rows.extend(
                    self._build_skipped_rows(
                        config=self.config,
                        task_name=task_name,
                        ks=ks,
                        details=f"task not supported for current config: {support_reason}",
                        retrieval_output=retrieval_output,
                    )
                )
                continue

            evaluator = self._create_task_evaluator(task_name)
            try:
                rows.extend(
                    evaluator.evaluate(
                        config=self.config,
                        retrieval_output=retrieval_output,
                        ks=ks,
                    )
                )
            except Exception as e:
                rows.extend(
                    self._build_skipped_rows(
                        config=self.config,
                        task_name=task_name,
                        ks=ks,
                        details=f"task failed at runtime: {e}",
                        retrieval_output=retrieval_output,
                    )
                )

        output_path = self._resolve_output_path(self.config, self.config_path)
        self._write_csv(rows, output_path)
        return {
            "rows": rows,
            "output_path": str(output_path.resolve()),
            "num_rows": len(rows),
        }

    @staticmethod
    def _normalize_task_name(task_name: str) -> str:
        return canonicalize_task_name(task_name)

    def _resolve_requested_tasks(self, config: dict[str, Any]) -> list[str]:
        evaluation_cfg = config.get("evaluation", {})
        requested_tasks = evaluation_cfg.get("extrinsic_tasks_to_run")
        if requested_tasks is None:
            requested_tasks = [evaluation_cfg.get("extrinsic_evaluator", "document_retrieval")]
        if not isinstance(requested_tasks, list) or not requested_tasks:
            raise ValueError("'evaluation.extrinsic_tasks_to_run' must be a non-empty list of task names.")

        normalized: list[str] = []
        seen: set[str] = set()
        for task_name in requested_tasks:
            if not isinstance(task_name, str) or not task_name.strip():
                raise ValueError("Each entry in 'evaluation.extrinsic_tasks_to_run' must be a non-empty string.")
            name = self._normalize_task_name(task_name)
            if name not in seen:
                seen.add(name)
                normalized.append(name)
        return normalized

    @staticmethod
    def _resolve_ks_for_task(config: dict[str, Any], task_name: str) -> list[int]:
        task_cfg = get_extrinsic_task_cfg(config, task_name)
        ks = task_cfg.get("ks", [1, 3, 5, 10])
        if not isinstance(ks, list) or not ks:
            raise ValueError(f"'evaluation.extrinsic_tasks.{task_name}.ks' must be a non-empty list.")
        resolved = sorted({int(k) for k in ks})
        if any(k <= 0 for k in resolved):
            raise ValueError(f"'evaluation.extrinsic_tasks.{task_name}.ks' must contain only positive integers.")
        return resolved

    def _task_support_status(self, config: dict[str, Any], task_name: str) -> tuple[bool, str]:
        if task_name == "document_retrieval":
            return check_document_retrieval_prerequisites(config)
        if task_name == "evidence_retrieval":
            return self._check_required_task_file(config=config, task_name=task_name)
        if task_name == "answer_generation":
            return self._check_required_task_file(config=config, task_name=task_name)
        return True, "no pre-check configured"

    @staticmethod
    def _check_required_task_file(*, config: dict[str, Any], task_name: str) -> tuple[bool, str]:
        if task_name == "evidence_retrieval":
            label = "evidence"
            resolver = resolve_evidence_path
            key = "evidence_path"
        else:
            label = "answers"
            resolver = resolve_answers_path
            key = "answers_path"
        try:
            value = resolver(config)
        except Exception as e:
            return False, f"missing {key}: {e}"
        if not value.exists():
            return False, f"{label} file does not exist: {value}"
        return True, f"{key}={value}"

    def _build_skipped_rows(
        self,
        *,
        config: dict[str, Any],
        task_name: str,
        ks: list[int],
        details: str,
        retrieval_output: dict[str, Any],
    ) -> list[dict[str, Any]]:
        context = build_run_context(
            config=config,
            task_name=task_name,
            default_experiment_name=task_name,
        )
        manifest_path = self._path_or_none(retrieval_output.get("manifest_path"))
        run_path = self._path_or_none(retrieval_output.get("run_path"))
        items_path = self._path_or_none(retrieval_output.get("items_path"))
        rows: list[dict[str, Any]] = []
        for k in ks:
            row = build_base_row(
                context=context,
                k=int(k),
                retrieval_manifest_path=manifest_path,
                retrieval_run_path=run_path,
                retrieval_items_path=items_path,
            )
            row.update({"status": "skipped", "details": details})
            rows.append(row)
        return rows

    @staticmethod
    def _resolve_output_path(config: dict[str, Any], config_path: Path | None) -> Path:
        dataset_name = config.get("dataset", {}).get("name", "unknown-dataset").replace("/", "-")
        chunking_name = config.get("chunking", {}).get("type", "unknown-chunking")
        router_cfg = config.get("router", {})
        routing_name = router_cfg.get("name", "no-routing") if router_cfg.get("enabled", False) else "no-routing"
        experiment_name = config.get("experiment_name") or config.get("experiment", {}).get("name") or "extrinsic-eval"
        config_name = config_path.stem.replace("/", "-") if config_path is not None else "default-run"
        output_dir = Path(config.get("evaluation", {}).get("results_dir", "results/extrinsic"))
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / f"{dataset_name}_{chunking_name}_{routing_name}_{experiment_name}_{config_name}.csv"

    @staticmethod
    def _write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
        if not rows:
            raise ValueError("No rows to write.")
        fieldnames: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
        with output_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({name: row.get(name) for name in fieldnames})

    @staticmethod
    def _path_or_none(value: Any) -> Path | None:
        if isinstance(value, str) and value.strip():
            return Path(value)
        return None

    @classmethod
    def _create_task_evaluator(cls, task_name: str) -> Any:
        normalized = cls._normalize_task_name(task_name)
        try:
            evaluator_cls = cls._TASKS[normalized]
        except KeyError as exc:
            raise ValueError(f"Unsupported extrinsic evaluator: {task_name}") from exc
        return evaluator_cls()
