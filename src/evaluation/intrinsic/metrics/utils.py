from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List


def normalize_chunk_output(chunk_output: Any) -> List[Dict[str, Any]]:
    """
    Normalize chunk output into:
    [
        {
            "doc_id": str,
            "chunks": [
                {
                    "chunk_id": str,
                    "doc_id": str,
                    "text": str,
                    "position": int,
                    "sentences": list[str],
                    "start_sentence_idx": int | None,
                    "end_sentence_idx": int | None,
                    "metadata": dict,
                },
                ...
            ],
        },
        ...
    ]

    Supported input formats:
    1) dict with key "chunks" returned by the chunker
    2) flat list of chunk dicts
    """
    raw_chunks = _extract_chunks(chunk_output)

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for chunk in raw_chunks:
        if not isinstance(chunk, dict):
            raise ValueError("Each chunk must be a dictionary.")

        required = {"chunk_id", "doc_id", "text", "position"}
        missing = required - set(chunk.keys())
        if missing:
            raise ValueError(f"Chunk missing required fields: {sorted(missing)}")

        normalized_chunk = {
            "chunk_id": str(chunk["chunk_id"]),
            "doc_id": str(chunk["doc_id"]),
            "text": str(chunk["text"]),
            "position": int(chunk["position"]),
            "sentences": list(chunk.get("sentences", [])),
            "start_sentence_idx": chunk.get("start_sentence_idx"),
            "end_sentence_idx": chunk.get("end_sentence_idx"),
            "metadata": dict(chunk.get("metadata", {})),
        }

        grouped[normalized_chunk["doc_id"]].append(normalized_chunk)

    documents: List[Dict[str, Any]] = []
    for doc_id, chunks in grouped.items():
        ordered_chunks = sorted(chunks, key=lambda x: x["position"])
        documents.append(
            {
                "doc_id": doc_id,
                "chunks": ordered_chunks,
            }
        )

    documents.sort(key=lambda x: x["doc_id"])
    return documents


def _extract_chunks(chunk_output: Any) -> List[Dict[str, Any]]:
    """
    Extract raw chunk list from the actual chunker output.
    """
    if isinstance(chunk_output, dict):
        if "chunks" not in chunk_output:
            raise ValueError(
                "Unsupported chunk_output dict format for intrinsic evaluation: "
                "missing 'chunks' key."
            )

        chunks = chunk_output["chunks"]
        if not isinstance(chunks, list):
            raise ValueError("'chunks' must be a list.")
        return chunks

    if isinstance(chunk_output, list):
        return chunk_output

    raise ValueError(
        "Unsupported chunk_output format for intrinsic evaluation: "
        "expected either a dict with key 'chunks' or a list of chunk dictionaries."
    )


def count_total_chunks(docs: List[Dict[str, Any]]) -> int:
    return sum(len(doc["chunks"]) for doc in docs)