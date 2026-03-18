from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

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
        default="yairfeldman/qasper",
        help="Hugging Face dataset id to read from.",
    )
    parser.add_argument(
        "--source-split",
        type=str,
        default="test",
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


def _load_qasper_dataset(hf_dataset: str, source_split: str):
    candidates: list[str] = [hf_dataset]
    if hf_dataset == "allenai/qasper":
        # Fallback mirrors compatible with datasets>=4 (no legacy dataset script).
        candidates.append("yairfeldman/qasper")
        candidates.append("kothasuhas/qasper")
    elif hf_dataset == "yairfeldman/qasper":
        candidates.append("kothasuhas/qasper")

    last_error: Exception | None = None
    for candidate in candidates:
        try:
            ds = load_dataset(candidate, split=source_split)
            return candidate, ds
        except RuntimeError as e:
            message = str(e)
            if "Dataset scripts are no longer supported" in message:
                print(
                    "[WARN] Dataset source uses legacy script and is not supported by current "
                    f"`datasets` version: {candidate}. Trying fallback source..."
                )
                last_error = e
                continue
            raise
        except Exception as e:
            last_error = e
            continue

    attempted = ", ".join(candidates)
    raise RuntimeError(
        "Could not load Qasper dataset from any source. "
        f"Attempted: {attempted}. Last error: {last_error}"
    )


def _safe_title(text: str, max_len: int = 120) -> str:
    head = " ".join(text.split())[:max_len].strip()
    return head or "Qasper document"


def _write_documents_jsonl(path: Path, documents: dict[str, dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for doc_id, payload in documents.items():
            row = {"id": doc_id, **payload}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _iter_strings(values: Iterable[Any]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = _stringify(value)
        if text:
            out.append(text)
    return out


def _unique_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _collect_evidence_texts(raw_answer: Any) -> list[str]:
    """
    Best-effort extraction across known Qasper schemas.
    """
    if raw_answer is None:
        return []

    if isinstance(raw_answer, str):
        text = raw_answer.strip()
        return [text] if text else []

    if isinstance(raw_answer, list):
        texts: list[str] = []
        for item in raw_answer:
            texts.extend(_collect_evidence_texts(item))
        return _unique_preserve_order(texts)

    if isinstance(raw_answer, dict):
        texts: list[str] = []

        # Common keys used for evidence spans/paragraphs.
        for key in ("evidence", "evidence_text", "evidence_texts", "supporting_facts"):
            value = raw_answer.get(key)
            if isinstance(value, str):
                if value.strip():
                    texts.append(value.strip())
            elif isinstance(value, list):
                texts.extend(_iter_strings(value))

        # Some variants nest evidence under "answer".
        nested_answer = raw_answer.get("answer")
        if nested_answer is not None:
            texts.extend(_collect_evidence_texts(nested_answer))

        return _unique_preserve_order(texts)

    return []


def _collect_reference_answers(raw_answer: Any) -> list[str]:
    """
    Best-effort extraction of textual answers across known Qasper schemas.
    """
    if raw_answer is None:
        return []

    if isinstance(raw_answer, str):
        text = raw_answer.strip()
        return [text] if text else []

    if isinstance(raw_answer, list):
        texts: list[str] = []
        for item in raw_answer:
            texts.extend(_collect_reference_answers(item))
        return _unique_preserve_order(texts)

    if isinstance(raw_answer, dict):
        texts: list[str] = []

        # Official Qasper style.
        for key in ("free_form_answer", "yes_no"):
            value = raw_answer.get(key)
            if isinstance(value, bool):
                texts.append("yes" if value else "no")
            elif isinstance(value, str) and value.strip():
                texts.append(value.strip())

        extractive = raw_answer.get("extractive_spans")
        if isinstance(extractive, list):
            texts.extend(_iter_strings(extractive))

        # Some wrappers nest under "answer".
        nested_answer = raw_answer.get("answer")
        if nested_answer is not None:
            texts.extend(_collect_reference_answers(nested_answer))

        return _unique_preserve_order(texts)

    return []


def _build_official_qasper_context(row: dict[str, Any]) -> str:
    title = _stringify(row.get("title"))
    abstract = _stringify(row.get("abstract"))

    full_text = row.get("full_text") or []
    section_texts: list[str] = []
    if isinstance(full_text, list):
        for section in full_text:
            if not isinstance(section, dict):
                continue
            section_name = _stringify(section.get("section_name"))
            paragraphs = section.get("paragraphs") or []
            paragraph_text = "\n".join(_iter_strings(paragraphs))
            if section_name and paragraph_text:
                section_texts.append(f"{section_name}\n{paragraph_text}")
            elif paragraph_text:
                section_texts.append(paragraph_text)
    elif isinstance(full_text, dict):
        section_names = full_text.get("section_name") or []
        paragraphs_by_section = full_text.get("paragraphs") or []
        if isinstance(section_names, list) and isinstance(paragraphs_by_section, list):
            n = min(len(section_names), len(paragraphs_by_section))
            for i in range(n):
                section_name = _stringify(section_names[i])
                paragraphs = paragraphs_by_section[i] if i < len(paragraphs_by_section) else []
                paragraph_text = "\n".join(_iter_strings(paragraphs if isinstance(paragraphs, list) else []))
                if section_name and paragraph_text:
                    section_texts.append(f"{section_name}\n{paragraph_text}")
                elif paragraph_text:
                    section_texts.append(paragraph_text)

    parts = [p for p in [title, abstract, "\n\n".join(section_texts)] if p]
    return "\n\n".join(parts).strip()


def main() -> None:
    args = parse_args()

    target_dir = Path(args.target_dir)
    qrels_dir = target_dir / "qrels"
    target_dir.mkdir(parents=True, exist_ok=True)
    qrels_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading HF dataset '{args.hf_dataset}' split='{args.source_split}'...")
    loaded_dataset_name, ds = _load_qasper_dataset(args.hf_dataset, args.source_split)
    if loaded_dataset_name != args.hf_dataset:
        print(f"[INFO] Using fallback HF dataset source: '{loaded_dataset_name}'")
    if args.max_documents is not None:
        ds = ds.select(range(min(args.max_documents, len(ds))))

    documents: dict[str, dict[str, Any]] = {}
    queries: dict[str, str] = {}
    qrels: dict[str, dict[str, int]] = {}
    evidences: dict[str, list[dict[str, Any]]] = {}
    answers: dict[str, dict[str, Any]] = {}

    for doc_idx, row in enumerate(ds):
        if not isinstance(row, dict):
            continue

        if "qas" in row:
            context = _build_official_qasper_context(row)
            raw_doc_id = _stringify(row.get("id"))
            doc_id = raw_doc_id or f"qasper-doc-{doc_idx:06d}"
        else:
            context = _stringify(row.get("context"))
            doc_id = f"qasper-doc-{doc_idx:06d}"

        if not context:
            continue
        if args.max_chars_per_doc is not None and args.max_chars_per_doc > 0 and len(context) > args.max_chars_per_doc:
            context = context[: args.max_chars_per_doc]

        documents[doc_id] = {
            "title": _safe_title(context),
            "text": context,
        }

        qas = row.get("qas")
        if isinstance(qas, list):
            qas_items = qas
            for q_idx, qa in enumerate(qas_items):
                if not isinstance(qa, dict):
                    continue
                q_text = _stringify(qa.get("question"))
                if not q_text:
                    continue
                query_id = f"{doc_id}-q{q_idx:03d}"
                queries[query_id] = q_text
                qrels[query_id] = {doc_id: 1}

                raw_answers = qa.get("answers") or []
                reference_answers = _collect_reference_answers(raw_answers)
                evidence_texts = _collect_evidence_texts(raw_answers)

                answers[query_id] = {
                    "doc_id": doc_id,
                    "reference_answers": reference_answers,
                    "answerable": bool(reference_answers),
                }
                evidences[query_id] = [
                    {
                        "doc_id": doc_id,
                        "evidence_text": text,
                    }
                    for text in evidence_texts
                ] or [{"doc_id": doc_id, "evidence_text": ""}]
            continue
        if isinstance(qas, dict):
            questions = qas.get("question") or []
            qas_answers = qas.get("answers") or []
            if isinstance(questions, list):
                for q_idx, question in enumerate(questions):
                    q_text = _stringify(question)
                    if not q_text:
                        continue
                    query_id = f"{doc_id}-q{q_idx:03d}"
                    queries[query_id] = q_text
                    qrels[query_id] = {doc_id: 1}

                    raw_answers = qas_answers[q_idx] if isinstance(qas_answers, list) and q_idx < len(qas_answers) else []
                    reference_answers = _collect_reference_answers(raw_answers)
                    evidence_texts = _collect_evidence_texts(raw_answers)

                    answers[query_id] = {
                        "doc_id": doc_id,
                        "reference_answers": reference_answers,
                        "answerable": bool(reference_answers),
                    }
                    evidences[query_id] = [
                        {
                            "doc_id": doc_id,
                            "evidence_text": text,
                        }
                        for text in evidence_texts
                    ] or [{"doc_id": doc_id, "evidence_text": ""}]
                continue

        # Fallback schema used by some mirrored HF versions.
        questions = row.get("questions") or []
        raw_answers = row.get("answers") or []
        raw_evidences = row.get("evidences") or []
        for q_idx, question in enumerate(questions):
            q_text = _stringify(question)
            if not q_text:
                continue
            query_id = f"{doc_id}-q{q_idx:03d}"
            queries[query_id] = q_text
            qrels[query_id] = {doc_id: 1}

            answer_values: list[str] = []
            if q_idx < len(raw_answers):
                answer_values = _collect_reference_answers(raw_answers[q_idx])

            evidence_values: list[str] = []
            if q_idx < len(raw_evidences):
                evidence_values = _collect_evidence_texts(raw_evidences[q_idx])

            answers[query_id] = {
                "doc_id": doc_id,
                "reference_answers": answer_values,
                "answerable": bool(answer_values),
            }
            evidences[query_id] = [
                {
                    "doc_id": doc_id,
                    "evidence_text": text,
                }
                for text in evidence_values
            ] or [{"doc_id": doc_id, "evidence_text": ""}]

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
        "source": f"huggingface:{loaded_dataset_name}:{args.source_split}",
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

    non_empty_evidence_queries = sum(
        1
        for annotations in evidences.values()
        if any(_stringify(item.get("evidence_text")) for item in annotations if isinstance(item, dict))
    )

    print(
        f"[OK] Normalized dataset written to {target_dir} "
        f"(docs={len(documents)}, queries={len(queries)}, qrels={len(qrels)})."
    )
    print(
        "[INFO] Evidence coverage: "
        f"queries_with_non_empty_evidence={non_empty_evidence_queries}/{len(queries)}"
    )


if __name__ == "__main__":
    main()
