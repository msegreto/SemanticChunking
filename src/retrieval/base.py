from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseRetriever(ABC):
    @abstractmethod
    def build_index(self, embedding_output: Any, config: Dict, global_config: Dict) -> Any:
        raise NotImplementedError

    @abstractmethod
    def load_index(self, index_path: str, metadata_path: str | None = None) -> Any:
        raise NotImplementedError

    @abstractmethod
    def search(self, query_vectors: Any, top_k: int = 10) -> Any:
        raise NotImplementedError