from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class EmbeddingItem:
    text: str
    item_id: Optional[str] = None
    doc_id: Optional[str] = None
    unit_id: Optional[int] = None
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

    if not isinstance(raw_items, list):
        raise TypeError(
            f"Embedding input collection must be a list. Got {type(raw_items).__name__}."
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

        raw_metadata = raw.get("metadata", {})
        if raw_metadata is None:
            raw_metadata = {}
        if not isinstance(raw_metadata, dict):
            raise TypeError(
                f"Item at index {idx} has invalid 'metadata': expected dict, "
                f"got {type(raw_metadata).__name__}."
            )

        item_id = raw.get("item_id", raw.get("id"))
        if item_id is not None:
            item_id = str(item_id)

        doc_id = raw.get("doc_id")
        if doc_id is not None:
            doc_id = str(doc_id)

        unit_id = raw.get("unit_id")
        if unit_id is not None:
            try:
                unit_id = int(unit_id)
            except (TypeError, ValueError) as e:
                raise TypeError(
                    f"Item at index {idx} has invalid 'unit_id': {unit_id!r}"
                ) from e

        item = EmbeddingItem(
            text=text.strip(),
            item_id=item_id,
            doc_id=doc_id,
            unit_id=unit_id,
            metadata=dict(raw_metadata),
        )

        # Salva eventuali campi aggiuntivi dentro metadata
        reserved_keys = {"text", "item_id", "id", "doc_id", "unit_id", "metadata"}
        extra_fields = {k: v for k, v in raw.items() if k not in reserved_keys}
        if extra_fields:
            item.metadata = {**item.metadata, **extra_fields}

        items.append(item)

    return items


def validate_embedder_config(
    config: dict[str, Any],
    *,
    allow_instruction: bool = False,
) -> dict[str, Any]:
    batch_size = config.get("batch_size", 32)
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("'batch_size' must be a positive integer.")

    normalize_embeddings = config.get("normalize_embeddings", True)
    if not isinstance(normalize_embeddings, bool):
        raise TypeError("'normalize_embeddings' must be a boolean.")

    show_progress_bar = config.get("show_progress_bar", False)
    if not isinstance(show_progress_bar, bool):
        raise TypeError("'show_progress_bar' must be a boolean.")

    convert_to_numpy = config.get("convert_to_numpy", True)
    if not isinstance(convert_to_numpy, bool):
        raise TypeError("'convert_to_numpy' must be a boolean.")

    validated = {
        "batch_size": batch_size,
        "normalize_embeddings": normalize_embeddings,
        "show_progress_bar": show_progress_bar,
        "convert_to_numpy": convert_to_numpy,
    }

    if allow_instruction:
        instruction = config.get("instruction", "")
        if not isinstance(instruction, str):
            raise TypeError("'instruction' must be a string.")
        validated["instruction"] = instruction

    return validated


def build_embedding_output(
    *,
    embeddings: Any,
    items: List[EmbeddingItem],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    output = {
        "embeddings": embeddings,
        "items": items,
        "metadata": metadata,
    }
    validate_embedding_output(output)
    return output


def validate_embedding_output(output: Any) -> dict[str, Any]:
    if not isinstance(output, dict):
        raise TypeError("Embedding output must be a dict.")

    required_keys = {"embeddings", "items", "metadata"}
    missing = required_keys - set(output.keys())
    if missing:
        raise ValueError(f"Embedding output missing required keys: {sorted(missing)}")

    items = output["items"]
    if not isinstance(items, list):
        raise TypeError("'items' in embedding output must be a list.")

    for idx, item in enumerate(items):
        if not isinstance(item, EmbeddingItem):
            raise TypeError(
                f"Embedding output item at index {idx} must be an EmbeddingItem. "
                f"Got {type(item).__name__}."
            )

    metadata = output["metadata"]
    if not isinstance(metadata, dict):
        raise TypeError("'metadata' in embedding output must be a dict.")

    embeddings = np.asarray(output["embeddings"])
    if embeddings.ndim != 2:
        raise ValueError(f"Embeddings must be a 2D array. Got shape: {embeddings.shape}")

    if embeddings.shape[0] != len(items):
        raise ValueError(
            "Embeddings row count must match number of items. "
            f"Got rows={embeddings.shape[0]} items={len(items)}."
        )

    if np.isnan(embeddings).any():
        raise ValueError("Embeddings contain NaN values.")

    return output


class BaseEmbedder(ABC):
    def encode_texts(self, texts: list[str], config: dict) -> tuple[Any, dict[str, Any]]:
        """
        API opzionale per riusare l'encoder a livello piu' basso.

        Serve ai casi in cui si vogliono vettorizzare testi senza costruire
        l'intero embedding output della fase retrieval.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement encode_texts()."
        )

    def encode_items(
        self,
        items: list[EmbeddingItem],
        config: dict,
    ) -> tuple[Any, list[EmbeddingItem], dict[str, Any]]:
        """
        API opzionale per codificare elementi gia' normalizzati.
        """
        texts = [item.text for item in items]
        embeddings, metadata = self.encode_texts(texts, config)
        return embeddings, items, metadata

    @abstractmethod
    def embed(self, data: Any, config: dict) -> dict[str, Any]:
        """
        Esegue embedding di unità testuali.

        L'input può rappresentare:
        - chunk finali
        - sentences
        - propositions
        - altre unità testuali

        Deve restituire un dict validato con:
        - embeddings
        - items normalizzati
        - metadata sul modello/configurazione
        """
        raise NotImplementedError
