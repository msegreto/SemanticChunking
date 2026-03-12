from __future__ import annotations

from pathlib import Path
from typing import Any

from src.datasets.base import BaseDatasetProcessor


class DummyDatasetProcessor(BaseDatasetProcessor):
    def ensure_available(self, config: dict) -> Path:
        return self.resolve_data_dir(config)

    def load(self, raw_path: Path, config: dict) -> Any:
        print(f"[DATASET] process() called with config={config}")
        return {
            "documents": [],
            "queries": [],
            "qrels": [],
            "metadata": {
                "dataset_name": config.get("name", "unknown"),
                "raw_path": str(raw_path),
                "processor": "dummy",
            },
        }