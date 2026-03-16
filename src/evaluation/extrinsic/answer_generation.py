from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from .base import BaseExtrinsicEvaluator
from .common import build_base_row, build_run_context


class AnswerGenerationEvaluator(BaseExtrinsicEvaluator):
    @property
    def task_name(self) -> str:
        return "answer_generation"

    def evaluate(
        self,
        *,
        config: dict[str, Any],
        index_path: Path,
        index_metadata_path: Path,
        ks: Sequence[int],
    ) -> list[dict[str, Any]]:
        task_cfg = config.get("evaluation", {}).get("extrinsic_tasks", {}).get(self.task_name, {})
        answers_file = self._resolve_optional_path(task_cfg.get("answers_path"))

        generation_model = task_cfg.get("generation_model", {})
        model_name = generation_model.get("name") if isinstance(generation_model, dict) else None
        model_name = model_name.strip() if isinstance(model_name, str) and model_name.strip() else None

        if not ks:
            return self._build_rows(
                config=config,
                ks=[None],
                index_path=index_path,
                index_metadata_path=index_metadata_path,
                status="skipped",
                details="Task skipped: empty ks.",
                answers_path=None,
                generation_model=None,
            )

        resolved_ks = sorted({int(k) for k in ks})

        if answers_file is None:
            return self._build_rows(
                config=config,
                ks=resolved_ks,
                index_path=index_path,
                index_metadata_path=index_metadata_path,
                status="skipped",
                details="Task skipped: missing answers_path in YAML.",
                answers_path=None,
                generation_model=model_name,
            )

        if not answers_file.exists():
            return self._build_rows(
                config=config,
                ks=resolved_ks,
                index_path=index_path,
                index_metadata_path=index_metadata_path,
                status="skipped",
                details=f"Task skipped: answer annotations file not found at {answers_file}.",
                answers_path=str(answers_file),
                generation_model=model_name,
            )

        if model_name is None:
            return self._build_rows(
                config=config,
                ks=resolved_ks,
                index_path=index_path,
                index_metadata_path=index_metadata_path,
                status="skipped",
                details="Task skipped: missing generation_model.name in YAML.",
                answers_path=str(answers_file),
                generation_model=None,
            )

        return self._build_rows(
            config=config,
            ks=resolved_ks,
            index_path=index_path,
            index_metadata_path=index_metadata_path,
            status="placeholder_not_implemented",
            details="Task wiring and validation are enabled; generation/scoring are not implemented yet.",
            answers_path=str(answers_file),
            generation_model=model_name,
        )

    @staticmethod
    def _resolve_optional_path(value: Any) -> Path | None:
        if isinstance(value, str) and value.strip():
            return Path(value)
        return None

    def _build_rows(
        self,
        *,
        config: dict[str, Any],
        ks: Sequence[int | None],
        index_path: Path,
        index_metadata_path: Path,
        status: str,
        details: str,
        answers_path: str | None,
        generation_model: str | None,
    ) -> list[dict[str, Any]]:
        context = build_run_context(
            config=config,
            task_name=self.task_name,
            default_experiment_name="answer-generation",
        )
        rows: list[dict[str, Any]] = []
        for k in ks:
            row = build_base_row(
                context=context,
                k=k,
                index_path=index_path,
                index_metadata_path=index_metadata_path,
            )
            row.update(
                {
                    "status": status,
                    "details": details,
                    "answers_path": answers_path,
                    "generation_model": generation_model,
                }
            )
            rows.append(row)
        return rows
