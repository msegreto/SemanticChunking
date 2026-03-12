from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, routed_output: Any, config: dict) -> Any:
        raise NotImplementedError