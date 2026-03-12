from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple


def _edge_weight(scorer: Any, q: str, d: str) -> float:
    """
    MoC-inspired edge:
        Edge(q, d) = (ppl(q) - ppl(q | d)) / ppl(q)

    Clamped to [0, 1].
    """
    ppl_q = scorer.perplexity(q)
    ppl_q_given_d = scorer.conditional_perplexity(
        target_text=q,
        context_text=d,
    )

    if ppl_q <= 0:
        return 0.0

    value = (ppl_q - ppl_q_given_d) / ppl_q
    return max(0.0, min(1.0, value))


def _structural_entropy(num_nodes: int, edges: List[Tuple[int, int, float]]) -> float:
    """
    Lightweight graph structural entropy approximation.

    Degree-based entropy:
        H = - sum_i p_i log2(p_i)
    where p_i is normalized node degree.

    Lower = tighter / stickier graph.
    """
    if num_nodes == 0:
        return 0.0

    degrees = [0.0 for _ in range(num_nodes)]
    total_weight = 0.0

    for i, j, w in edges:
        degrees[i] += w
        degrees[j] += w
        total_weight += w

    total_degree = sum(degrees)
    if total_degree <= 0:
        return 0.0

    entropy = 0.0
    for degree in degrees:
        if degree <= 0:
            continue
        p = degree / total_degree
        entropy -= p * math.log2(p)

    return entropy


def compute_chunk_stickiness(
    doc: Dict[str, Any],
    scorer: Any,
    config: Dict[str, Any],
) -> Dict[str, float]:
    """
    Compute two variants:
    - complete: all chunk pairs
    - incomplete: local-window pairs only
    """
    chunks: List[Dict[str, Any]] = doc["chunks"]
    n = len(chunks)

    if n < 2:
        return {"complete": 0.0, "incomplete": 0.0}

    model_cfg = config.get("intrinsic_model", {})
    edge_threshold = float(model_cfg.get("edge_threshold", 0.0))
    local_window = int(model_cfg.get("local_window", 1))

    texts = [chunk["text"] for chunk in chunks]

    complete_edges: List[Tuple[int, int, float]] = []
    incomplete_edges: List[Tuple[int, int, float]] = []

    for i in range(n):
        for j in range(i + 1, n):
            w_ij = _edge_weight(scorer, texts[i], texts[j])
            w_ji = _edge_weight(scorer, texts[j], texts[i])
            w = max(w_ij, w_ji)

            if w < edge_threshold:
                continue

            complete_edges.append((i, j, w))

            if abs(i - j) <= local_window:
                incomplete_edges.append((i, j, w))

    return {
        "complete": _structural_entropy(n, complete_edges),
        "incomplete": _structural_entropy(n, incomplete_edges),
    }