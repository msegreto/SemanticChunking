from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseSplitter(ABC):
    def try_load_reusable_output(self, dataset_output: Any, config: dict) -> Any | None:
        return None

    def save_output(self, split_output: Any, config: dict) -> None:
        return None

    def resolve_output_path(self, config: dict) -> Path | None:
        output_path = config.get("output_path")
        return Path(output_path) if output_path else None

    @abstractmethod
    def split(self, dataset_output: Any, config: dict) -> Any:
        raise NotImplementedError
