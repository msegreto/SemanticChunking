from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseRouter(ABC):
    @abstractmethod
    def route(self, split_output: Any, config: dict) -> Any:
        raise NotImplementedError