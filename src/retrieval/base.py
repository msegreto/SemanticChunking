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

    def start_incremental_index(self, config: Dict, global_config: Dict) -> None:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support incremental indexing."
        )

    def add_embeddings_batch(
        self,
        vectors: Any,
        items: list[dict[str, Any]],
        embedding_metadata: Dict[str, Any] | None = None,
    ) -> None:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support incremental indexing."
        )

    def finalize_incremental_index(self) -> Dict[str, Any]:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support incremental indexing."
        )
