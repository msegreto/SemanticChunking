from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from .base import BaseExtrinsicEvaluator
from .common import build_base_row, build_run_context


class EvidenceRetrievalEvaluator(BaseExtrinsicEvaluator):
    @property
    def task_name(self) -> str:
        return "evidence_retrieval"

    def evaluate(
        self,
        *,
        config: dict[str, Any],
        index_path: Path,
        index_metadata_path: Path,
        ks: Sequence[int],
    ) -> list[dict[str, Any]]:
        task_cfg = config.get("evaluation", {}).get("extrinsic_tasks", {}).get(self.task_name, {})
        evidence_file = self._resolve_optional_path(task_cfg.get("evidence_path"))

        if not ks:
            return self._build_rows(
                config=config,
                ks=[None],
                index_path=index_path,
                index_metadata_path=index_metadata_path,
                status="skipped",
                details="Task skipped: empty ks.",
                evidence_path=None,
            )

        resolved_ks = sorted({int(k) for k in ks})

        if evidence_file is None:
            return self._build_rows(
                config=config,
                ks=resolved_ks,
                index_path=index_path,
                index_metadata_path=index_metadata_path,
                status="skipped",
                details="Task skipped: missing evidence_path in YAML.",
                evidence_path=None,
            )

        if not evidence_file.exists():
            return self._build_rows(
                config=config,
                ks=resolved_ks,
                index_path=index_path,
                index_metadata_path=index_metadata_path,
                status="skipped",
                details=f"Task skipped: evidence annotations file not found at {evidence_file}.",
                evidence_path=str(evidence_file),
            )

        return self._build_rows(
            config=config,
            ks=resolved_ks,
            index_path=index_path,
            index_metadata_path=index_metadata_path,
            status="placeholder_not_implemented",
            details="Task wiring and validation are enabled; scoring is not implemented yet.",
            evidence_path=str(evidence_file),
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
        evidence_path: str | None,
    ) -> list[dict[str, Any]]:
        context = build_run_context(
            config=config,
            task_name=self.task_name,
            default_experiment_name="evidence-retrieval",
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
                    "evidence_path": evidence_path,
                }
            )
            rows.append(row)
        return rows
