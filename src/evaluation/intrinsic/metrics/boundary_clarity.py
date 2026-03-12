from __future__ import annotations

from typing import Any, Dict, List


def compute_boundary_clarity(
    doc: Dict[str, Any],
    scorer: Any,
    config: Dict[str, Any],
) -> float:
    """
    MoC-inspired Boundary Clarity.

    For each boundary between chunk_i and chunk_{i+1}, compute:
        BC_i = ppl(right | left) / ppl(right)

    Higher means the right chunk remains relatively unpredictable
    even when conditioned on the left chunk, so the boundary is clearer.
    """
    chunks: List[Dict[str, Any]] = doc["chunks"]

    if len(chunks) < 2:
        return 1.0

    values: List[float] = []

    for i in range(len(chunks) - 1):
        left_text = chunks[i]["text"]
        right_text = chunks[i + 1]["text"]

        ppl_right = scorer.perplexity(right_text)
        ppl_right_given_left = scorer.conditional_perplexity(
            target_text=right_text,
            context_text=left_text,
        )

        if ppl_right <= 0:
            continue

        bc = ppl_right_given_left / ppl_right
        values.append(bc)

    if not values:
        return 1.0

    return sum(values) / len(values)