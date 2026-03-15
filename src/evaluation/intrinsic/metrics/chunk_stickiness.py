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
    Structural entropy aligned with the MoC paper:
        CS(G) = -sum_i (h_i / (2m)) * log2(h_i / (2m))
    where h_i is node degree and m is number of edges.
    """
    if num_nodes == 0:
        return 0.0

    degrees = [0.0 for _ in range(num_nodes)]

    for i, j, _w in edges:
        degrees[i] += 1.0
        degrees[j] += 1.0

    m = len(edges)
    if m <= 0:
        return 0.0

    normalizer = 2.0 * float(m)
    entropy = 0.0
    for degree in degrees:
        if degree <= 0:
            continue
        p = degree / normalizer
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
    - incomplete: order-constrained pairs only
    """
    chunks: List[Dict[str, Any]] = doc["chunks"]
    n = len(chunks)

    if n < 2:
        return {
            "complete": 0.0,
            "incomplete": 0.0,
            "complete_edges": 0.0,
            "incomplete_edges": 0.0,
        }

    model_cfg = config.get("intrinsic_model", {})
    edge_threshold = float(model_cfg.get("edge_threshold", 0.8))
    edge_threshold = max(0.0, min(1.0, edge_threshold))
    sequential_delta = int(model_cfg.get("sequential_delta", 0))

    texts = [chunk["text"] for chunk in chunks]

    complete_edges: List[Tuple[int, int, float]] = []
    incomplete_edges: List[Tuple[int, int, float]] = []

    for i in range(n):
        for j in range(i + 1, n):
            # Paper formulation uses a directed relevance criterion Edge(d_i, d_j) > K.
            w = _edge_weight(scorer, texts[i], texts[j])

            if w < edge_threshold:
                continue

            complete_edges.append((i, j, w))

            # Strict-paper style sequence-aware constraint.
            if (j - i) > sequential_delta:
                incomplete_edges.append((i, j, w))

    return {
        "complete": _structural_entropy(n, complete_edges),
        "incomplete": _structural_entropy(n, incomplete_edges),
        "complete_edges": float(len(complete_edges)),
        "incomplete_edges": float(len(incomplete_edges)),
    }
