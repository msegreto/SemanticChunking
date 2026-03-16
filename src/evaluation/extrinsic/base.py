from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Sequence


class BaseExtrinsicEvaluator(ABC):
    @property
    @abstractmethod
    def task_name(self) -> str:
        """Unique extrinsic task identifier (e.g. document_retrieval)."""

    @abstractmethod
    def evaluate(
        self,
        *,
        config: Dict[str, Any],
        index_path: Path,
        index_metadata_path: Path,
        ks: Sequence[int],
    ) -> list[dict[str, Any]]:
        """Run the extrinsic evaluation and return row dicts ready for CSV export."""
