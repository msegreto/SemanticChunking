from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class EmbeddingItem:
    text: str
    item_id: Optional[str] = None
    doc_id: Optional[str] = None
    unit_id: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddingResult:
    embeddings: Any
    items: List[EmbeddingItem]
    metadata: Dict[str, Any] = field(default_factory=dict)


def normalize_items(data: Any) -> List[EmbeddingItem]:
    """
    Normalizza diversi possibili formati di input in una lista di EmbeddingItem.

    Formati supportati:
    - list[dict] con almeno la chiave "text"
    - {"chunks": [...]}
    - {"sentences": [...]}
    - {"units": [...]}
    - {"items": [...]}

    Ogni elemento può avere campi come:
    - text
    - id / item_id
    - doc_id
    - unit_id
    - metadata
    """
    if data is None:
        raise ValueError("Input data for embedding is None.")

    if isinstance(data, dict):
        if "items" in data:
            raw_items = data["items"]
        elif "chunks" in data:
            raw_items = data["chunks"]
        elif "sentences" in data:
            raw_items = data["sentences"]
        elif "units" in data:
            raw_items = data["units"]
        else:
            available = ", ".join(data.keys())
            raise ValueError(
                "Unsupported dict input format for embedding. "
                f"Expected one of: items, chunks, sentences, units. Got keys: {available}"
            )
    elif isinstance(data, list):
        raw_items = data
    else:
        raise TypeError(
            "Embedding input must be either a list of items or a dict containing "
            "'items', 'chunks', 'sentences' or 'units'."
        )

    items: List[EmbeddingItem] = []

    for idx, raw in enumerate(raw_items):
        if not isinstance(raw, dict):
            raise TypeError(
                f"Each embedding item must be a dict. Got {type(raw).__name__} at index {idx}."
            )

        text = raw.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                f"Item at index {idx} is missing a valid non-empty 'text' field."
            )

        item = EmbeddingItem(
            text=text,
            item_id=raw.get("item_id", raw.get("id")),
            doc_id=raw.get("doc_id"),
            unit_id=raw.get("unit_id"),
            metadata=raw.get("metadata", {}),
        )

        # Salva eventuali campi aggiuntivi dentro metadata
        reserved_keys = {"text", "item_id", "id", "doc_id", "unit_id", "metadata"}
        extra_fields = {k: v for k, v in raw.items() if k not in reserved_keys}
        if extra_fields:
            item.metadata = {**item.metadata, **extra_fields}

        items.append(item)

    return items


class BaseEmbedder(ABC):
    @abstractmethod
    def embed(self, data: Any, config: dict) -> EmbeddingResult:
        """
        Esegue embedding di unità testuali.

        L'input può rappresentare:
        - chunk finali
        - sentences
        - propositions
        - altre unità testuali

        Deve restituire un EmbeddingResult con:
        - embeddings
        - items normalizzati
        - metadata sul modello/configurazione
        """
        raise NotImplementedError