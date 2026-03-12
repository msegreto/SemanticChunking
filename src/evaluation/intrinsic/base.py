from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseIntrinsicEvaluator(ABC):
    @abstractmethod
    def evaluate(self, chunk_output: Any, config: dict) -> Any:
        raise NotImplementedError