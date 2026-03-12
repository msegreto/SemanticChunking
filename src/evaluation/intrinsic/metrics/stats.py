from __future__ import annotations

from typing import Any, Dict, List


def build_global_statistics(results: Dict[str, Any]) -> Dict[str, Any]:
    per_doc = results.get("per_document_metrics", [])

    metric_names = _extract_numeric_metric_names(per_doc)

    summary: Dict[str, Any] = {}
    for metric_name in metric_names:
        values = [
            float(doc[metric_name])
            for doc in per_doc
            if metric_name in doc and isinstance(doc[metric_name], (int, float))
        ]

        if not values:
            continue

        mean_value = sum(values) / len(values)
        variance_value = sum((x - mean_value) ** 2 for x in values) / len(values)

        summary[metric_name] = {
            "mean": mean_value,
            "min": min(values),
            "max": max(values),
            "variance": variance_value,
        }

    summary["metadata"] = {
        "num_documents": len(per_doc),
    }

    return summary


def _extract_numeric_metric_names(per_doc: List[Dict[str, Any]]) -> List[str]:
    excluded = {"doc_id", "num_chunks"}
    names = set()

    for doc in per_doc:
        for key, value in doc.items():
            if key in excluded:
                continue
            if isinstance(value, (int, float)):
                names.add(key)

    return sorted(names)