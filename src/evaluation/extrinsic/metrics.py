from __future__ import annotations


def precision_recall_f1(pred_count: int, gold_count: int, tp_count: int) -> tuple[float, float, float]:
    precision = tp_count / pred_count if pred_count > 0 else 0.0
    recall = tp_count / gold_count if gold_count > 0 else 0.0
    if precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    return precision, recall, f1