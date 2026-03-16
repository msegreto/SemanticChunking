from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download Qasper from Hugging Face and write a normalized dataset "
            "compatible with this pipeline under data/normalized/."
        )
    )
    parser.add_argument(
        "--hf-dataset",
        type=str,
        default="hulki/allenai_qasper",
        help="Hugging Face dataset id to read from.",
    )
    parser.add_argument(
        "--source-split",
        type=str,
        default="validation",
        help="Source split to read from the HF dataset (e.g. train/validation).",
    )
    parser.add_argument(
        "--output-split",
        type=str,
        default="test",
        help="Target split name written under qrels/<split>.json.",
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        default="data/normalized/beir/qasper",
        help="Target normalized directory.",
    )
    parser.add_argument(
        "--max-documents",
        type=int,
        default=None,
        help="Optional cap for debug runs.",
    )
    parser.add_argument(
        "--max-chars-per-doc",
        type=int,
        default=900000,
        help="Trim overly long contexts to avoid spaCy max_length failures in sentence splitting.",
    )
    return parser.parse_args()


def _safe_title(text: str, max_len: int = 120) -> str:
    head = " ".join(text.split())[:max_len].strip()
    return head or "Qasper document"


def _write_documents_jsonl(path: Path, documents: dict[str, dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for doc_id, payload in documents.items():
            row = {"id": doc_id, **payload}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()

    target_dir = Path(args.target_dir)
    qrels_dir = target_dir / "qrels"
    target_dir.mkdir(parents=True, exist_ok=True)
    qrels_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[INFO] Loading HF dataset '{args.hf_dataset}' split='{args.source_split}'..."
    )
    ds = load_dataset(args.hf_dataset, split=args.source_split)
    if args.max_documents is not None:
        ds = ds.select(range(min(args.max_documents, len(ds))))

    documents: dict[str, dict[str, Any]] = {}
    queries: dict[str, str] = {}
    qrels: dict[str, dict[str, int]] = {}
    evidences: dict[str, list[dict[str, Any]]] = {}
    answers: dict[str, dict[str, Any]] = {}

    for doc_idx, row in enumerate(ds):
        context = str(row.get("context", "")).strip()
        if not context:
            continue
        if args.max_chars_per_doc is not None and args.max_chars_per_doc > 0 and len(context) > args.max_chars_per_doc:
            context = context[: args.max_chars_per_doc]

        doc_id = f"qasper-doc-{doc_idx:06d}"
        documents[doc_id] = {
            "title": _safe_title(context),
            "text": context,
        }

        questions = row.get("questions") or []
        raw_answers = row.get("answers") or []
        for q_idx, question in enumerate(questions):
            q_text = str(question).strip()
            if not q_text:
                continue
            query_id = f"{doc_id}-q{q_idx:03d}"
            queries[query_id] = q_text
            qrels[query_id] = {doc_id: 1}

            answer_values: list[str] = []
            if q_idx < len(raw_answers):
                raw_item = raw_answers[q_idx]
                if isinstance(raw_item, list):
                    answer_values = [str(x).strip() for x in raw_item if str(x).strip()]
                elif isinstance(raw_item, str):
                    cleaned = raw_item.strip()
                    if cleaned:
                        answer_values = [cleaned]

            answers[query_id] = {
                "doc_id": doc_id,
                "reference_answers": answer_values,
                "answerable": bool(answer_values),
            }
            evidences[query_id] = [
                {
                    "doc_id": doc_id,
                    "evidence_text": answer_values[0] if answer_values else "",
                }
            ]

    if not documents:
        raise RuntimeError("No documents were created from source dataset.")
    if not queries:
        raise RuntimeError("No queries were created from source dataset.")

    documents_path = target_dir / "documents.jsonl"
    queries_path = target_dir / "queries.json"
    qrels_path = qrels_dir / f"{args.output_split}.json"
    evidences_path = target_dir / "evidences.json"
    answers_path = target_dir / "answers.json"
    metadata_path = target_dir / "metadata.json"

    _write_documents_jsonl(documents_path, documents)
    queries_path.write_text(json.dumps(queries, ensure_ascii=False, indent=2), encoding="utf-8")
    qrels_path.write_text(json.dumps(qrels, ensure_ascii=False, indent=2), encoding="utf-8")
    evidences_path.write_text(json.dumps(evidences, ensure_ascii=False, indent=2), encoding="utf-8")
    answers_path.write_text(json.dumps(answers, ensure_ascii=False, indent=2), encoding="utf-8")

    metadata = {
        "dataset_name": "beir/qasper",
        "processor": "qasper_hf_converter",
        "split": args.output_split,
        "source": f"huggingface:{args.hf_dataset}:{args.source_split}",
        "num_documents": len(documents),
        "num_queries": len(queries),
        "num_qrels_queries": len(qrels),
        "normalized_schema_version": "1.0",
        "normalized_files": {
            "documents": "documents.jsonl",
            "queries": "queries.json",
            "qrels": f"qrels/{args.output_split}.json",
            "evidences": "evidences.json",
            "answers": "answers.json",
        },
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"[OK] Normalized dataset written to {target_dir} "
        f"(docs={len(documents)}, queries={len(queries)}, qrels={len(qrels)})."
    )


if __name__ == "__main__":
    main()
