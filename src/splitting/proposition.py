from __future__ import annotations

from typing import Any

from src.splitting.base import BaseSplitter


class PropositionSplitter(BaseSplitter):
    def split(self, dataset_output: Any, config: dict) -> Any:
        print(f"[SPLIT] proposition split with config={config}")
        return {
            "split_units": [],
            "metadata": {
                "split_type": "proposition",
                "source_metadata": dataset_output.get("metadata", {}),
            },
        }