from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseSplitter(ABC):
    @abstractmethod
    def split(self, dataset_output: Any, config: dict) -> Any:
        raise NotImplementedError