from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path
from typing import Any, Iterable

from datasets import get_dataset_config_names, load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download TechQA from Hugging Face and write a normalized dataset "
            "compatible with this pipeline under data/normalized/."
        )
    )
    parser.add_argument(
        "--hf-dataset",
        type=str,
        default="PrimeQA/TechQA",
        help="Hugging Face dataset id to read from.",
    )
    parser.add_argument(
        "--docs-config",
        type=str,
        default=None,
        help="Optional dataset config for documents (3-config mode).",
    )
    parser.add_argument(
        "--queries-config",
        type=str,
        default=None,
        help="Optional dataset config for queries (3-config mode).",
    )
    parser.add_argument(
        "--qrels-config",
        type=str,
        default=None,
        help="Optional dataset config for qrels (3-config mode).",
    )
    parser.add_argument(
        "--source-split",
        type=str,
        default="test",
        help="Source split to read from HF dataset (e.g. train/dev/test/validation).",
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
        default="data/normalized/beir/techqa",
        help="Target normalized directory.",
    )
    parser.add_argument(
        "--max-documents",
        type=int,
        default=None,
        help="Optional cap for debug runs.",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Optional cap for debug runs.",
    )
    parser.add_argument(
        "--max-chars-per-doc",
        type=int,
        default=900000,
        help="Trim overly long contexts to avoid sentence-splitting max_length failures.",
    )
    return parser.parse_args()


def _safe_title(text: str, max_len: int = 120) -> str:
    head = " ".join(text.split())[:max_len].strip()
    return head or "TechQA document"


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _unique_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if not value:
            continue
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _write_documents_jsonl(path: Path, documents: dict[str, dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for doc_id, payload in documents.items():
            row = {"id": doc_id, **payload}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _first_non_empty(row: dict[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        text = _stringify(row.get(key))
        if text:
            return text
    return ""


def _collect_text_values(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        text = raw.strip()
        return [text] if text else []
    if isinstance(raw, (int, float, bool)):
        return [str(raw)]
    if isinstance(raw, list):
        out: list[str] = []
        for item in raw:
            out.extend(_collect_text_values(item))
        return _unique_preserve_order(out)
    if isinstance(raw, dict):
        out: list[str] = []
        for key in (
            "text",
            "answer",
            "answers",
            "reference_answers",
            "extractive_spans",
            "evidence",
            "evidences",
            "supporting_facts",
        ):
            if key in raw:
                out.extend(_collect_text_values(raw.get(key)))
        return _unique_preserve_order(out)
    return []


def _collect_reference_answers(row: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key in ("answers", "answer", "reference_answers", "gold_answers", "final_answers"):
        values.extend(_collect_text_values(row.get(key)))
    yes_no = row.get("yes_no")
    if isinstance(yes_no, bool):
        values.append("yes" if yes_no else "no")
    return _unique_preserve_order(values)


def _collect_evidence_texts(row: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key in ("evidence", "evidences", "supporting_facts", "supporting_sentences"):
        values.extend(_collect_text_values(row.get(key)))
    return _unique_preserve_order(values)


def _load_split(hf_dataset: str, split_name: str, config_name: str | None = None):
    try:
        if config_name:
            return load_dataset(hf_dataset, config_name, split=split_name)
        return load_dataset(hf_dataset, split=split_name)
    except ValueError as e:
        if config_name and "BuilderConfig" in str(e):
            try:
                available = get_dataset_config_names(hf_dataset)
            except Exception:
                available = []
            raise ValueError(
                f"Config '{config_name}' not found for dataset '{hf_dataset}'. "
                f"Available configs: {available}"
            ) from e
        raise


def _normalize_doc_text(text: str, max_chars_per_doc: int | None) -> str:
    text = text.strip()
    if not text:
        return ""
    if max_chars_per_doc is not None and max_chars_per_doc > 0 and len(text) > max_chars_per_doc:
        return text[:max_chars_per_doc]
    return text


def _context_items_from_row(row: dict[str, Any]) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []

    # common list-style context containers
    for key in ("contexts", "documents", "docs", "retrieved_contexts", "passages"):
        value = row.get(key)
        if isinstance(value, list):
            for idx, item in enumerate(value):
                if isinstance(item, dict):
                    doc_id = _first_non_empty(item, ["doc_id", "id", "corpus_id", "passage_id", "pid"])
                    title = _first_non_empty(item, ["title", "doc_title"])
                    text = _first_non_empty(item, ["text", "document", "context", "content", "passage"])
                else:
                    doc_id = ""
                    title = ""
                    text = _stringify(item)
                if not text:
                    continue
                items.append({"doc_id": doc_id or f"ctx-{idx:06d}", "title": title, "text": text})

    # single context string fallback
    if not items:
        text = _first_non_empty(row, ["context", "document", "passage", "text_context"])
        if text:
            items.append(
                {
                    "doc_id": _first_non_empty(row, ["doc_id", "document_id", "corpus_id"]) or "",
                    "title": _first_non_empty(row, ["title", "doc_title"]),
                    "text": text,
                }
            )

    return items


def _build_from_three_config_mode(
    docs_ds: Any,
    queries_ds: Any,
    qrels_ds: Any,
    max_chars_per_doc: int | None,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, str],
    dict[str, dict[str, int]],
    dict[str, list[dict[str, Any]]],
    dict[str, dict[str, Any]],
    bool,
]:
    documents: dict[str, dict[str, Any]] = {}
    passage_text_by_id: dict[str, str] = {}

    for row_idx, row in enumerate(docs_ds):
        if not isinstance(row, dict):
            continue

        doc_id = _first_non_empty(row, ["doc_id", "id", "document_id", "_id"]) or f"techqa-doc-{row_idx:06d}"
        title = _first_non_empty(row, ["title", "doc_title", "document_title"])

        passage_ids = row.get("passage_ids")
        passages = row.get("passages")
        if isinstance(passage_ids, list) and isinstance(passages, list) and passage_ids:
            for p_idx, raw_pid in enumerate(passage_ids):
                passage_id = _stringify(raw_pid)
                if not passage_id:
                    continue
                passage_text = _stringify(passages[p_idx]) if p_idx < len(passages) else ""
                passage_text = _normalize_doc_text(passage_text, max_chars_per_doc)
                if not passage_text:
                    continue
                documents[passage_id] = {
                    "title": title or _safe_title(passage_text),
                    "text": passage_text,
                }
                passage_text_by_id[passage_id] = passage_text
            continue

        text = _first_non_empty(row, ["text", "document", "doc_text", "context", "contents"])
        text = _normalize_doc_text(text, max_chars_per_doc)
        if not text:
            continue

        documents[doc_id] = {"title": title or _safe_title(text), "text": text}
        passage_text_by_id[doc_id] = text

    queries: dict[str, str] = {}
    answers: dict[str, dict[str, Any]] = {}
    evidences: dict[str, list[dict[str, Any]]] = {}
    source_has_answer_fields = False

    for row_idx, row in enumerate(queries_ds):
        if not isinstance(row, dict):
            continue

        source_has_answer_fields = source_has_answer_fields or any(
            key in row
            for key in (
                "answers",
                "answer",
                "reference_answers",
                "gold_answers",
                "final_answers",
                "yes_no",
            )
        )

        query_id = _first_non_empty(row, ["query_id", "id", "question_id", "_id"]) or f"techqa-q-{row_idx:06d}"
        question = _first_non_empty(row, ["query", "question", "text"])
        if not question:
            continue

        queries[query_id] = question
        linked_doc_id = _first_non_empty(row, ["doc_id", "document_id", "corpus_id"])
        ref_answers = _collect_reference_answers(row)
        evidence_texts = _collect_evidence_texts(row)

        answers[query_id] = {
            "doc_id": linked_doc_id,
            "reference_answers": ref_answers,
            "answerable": bool(ref_answers),
        }
        evidences[query_id] = [
            {"doc_id": linked_doc_id, "evidence_text": text}
            for text in evidence_texts
        ] or [{"doc_id": linked_doc_id, "evidence_text": ""}]

    qrels: dict[str, dict[str, int]] = {}
    for row in qrels_ds:
        if not isinstance(row, dict):
            continue

        query_id = _first_non_empty(row, ["query_id", "qid", "question_id", "id"])
        doc_id = _first_non_empty(row, ["corpus_id", "doc_id", "document_id", "pid"])
        if not query_id or not doc_id:
            continue

        score_raw = row.get("score", row.get("relevance", row.get("label", 1)))
        try:
            score = int(score_raw)
        except (TypeError, ValueError):
            score = 1
        if score <= 0:
            continue

        qrels.setdefault(query_id, {})[doc_id] = score

    if qrels:
        allowed = set(qrels.keys())
        queries = {qid: qtext for qid, qtext in queries.items() if qid in allowed}
        answers = {qid: payload for qid, payload in answers.items() if qid in allowed}
        evidences = {qid: payload for qid, payload in evidences.items() if qid in allowed}

    for query_id in queries:
        answers.setdefault(query_id, {"doc_id": "", "reference_answers": [], "answerable": False})
        evidences.setdefault(query_id, [{"doc_id": "", "evidence_text": ""}])

        top_doc_id = ""
        if query_id in qrels and qrels[query_id]:
            top_doc_id = next(iter(qrels[query_id].keys()))
        if top_doc_id and not _stringify(answers[query_id].get("doc_id")):
            answers[query_id]["doc_id"] = top_doc_id

        top_text = passage_text_by_id.get(top_doc_id, "")
        if top_text:
            has_non_empty = any(
                _stringify(item.get("evidence_text"))
                for item in evidences[query_id]
                if isinstance(item, dict)
            )
            if not has_non_empty:
                evidences[query_id] = [{"doc_id": top_doc_id, "evidence_text": top_text}]

    return documents, queries, qrels, evidences, answers, source_has_answer_fields


def _build_from_single_dataset_mode(
    qa_ds: Any,
    max_chars_per_doc: int | None,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, str],
    dict[str, dict[str, int]],
    dict[str, list[dict[str, Any]]],
    dict[str, dict[str, Any]],
    bool,
]:
    documents: dict[str, dict[str, Any]] = {}
    queries: dict[str, str] = {}
    qrels: dict[str, dict[str, int]] = {}
    evidences: dict[str, list[dict[str, Any]]] = {}
    answers: dict[str, dict[str, Any]] = {}
    source_has_answer_fields = False

    for row_idx, row in enumerate(qa_ds):
        if not isinstance(row, dict):
            continue

        query_id = _first_non_empty(row, ["query_id", "question_id", "id", "_id"]) or f"techqa-q-{row_idx:06d}"
        question = _first_non_empty(row, ["query", "question", "text"])
        if not question:
            continue

        queries[query_id] = question
        reference_answers = _collect_reference_answers(row)
        source_has_answer_fields = source_has_answer_fields or any(
            key in row for key in ("answers", "answer", "reference_answers", "gold_answers", "final_answers", "yes_no")
        )

        context_items = _context_items_from_row(row)
        qrel_entries: dict[str, int] = {}
        evidence_items: list[dict[str, Any]] = []

        for item in context_items:
            text = _normalize_doc_text(item["text"], max_chars_per_doc)
            if not text:
                continue
            raw_doc_id = _stringify(item.get("doc_id"))
            if not raw_doc_id or raw_doc_id.startswith("ctx-"):
                digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
                raw_doc_id = f"techqa-doc-{digest}"

            if raw_doc_id not in documents:
                documents[raw_doc_id] = {
                    "title": _stringify(item.get("title")) or _safe_title(text),
                    "text": text,
                }

            qrel_entries[raw_doc_id] = 1
            evidence_items.append({"doc_id": raw_doc_id, "evidence_text": text})

        if qrel_entries:
            qrels[query_id] = qrel_entries

        evidence_texts = _collect_evidence_texts(row)
        if evidence_texts:
            evidence_items = [
                {"doc_id": next(iter(qrel_entries.keys())) if qrel_entries else "", "evidence_text": ev}
                for ev in evidence_texts
            ]

        evidences[query_id] = evidence_items or [{"doc_id": "", "evidence_text": ""}]
        answers[query_id] = {
            "doc_id": next(iter(qrel_entries.keys())) if qrel_entries else "",
            "reference_answers": reference_answers,
            "answerable": bool(reference_answers),
        }

    return documents, queries, qrels, evidences, answers, source_has_answer_fields


def _resolve_primeqa_techqa_tar() -> Path:
    base = Path.home() / ".cache" / "huggingface" / "hub" / "datasets--PrimeQA--TechQA"
    refs_main = base / "refs" / "main"
    if refs_main.exists():
        sha = refs_main.read_text(encoding="utf-8").strip()
        candidate = base / "snapshots" / sha / "TechQA.tar.gz"
        if candidate.exists():
            return candidate

    snapshots = base / "snapshots"
    if snapshots.exists():
        for tar_path in sorted(snapshots.glob("*/TechQA.tar.gz"), reverse=True):
            if tar_path.exists():
                return tar_path

    raise FileNotFoundError(
        "Could not locate PrimeQA/TechQA tar archive in local HF cache. "
        "Expected path under ~/.cache/huggingface/hub/datasets--PrimeQA--TechQA/snapshots/*/TechQA.tar.gz"
    )


def _build_from_primeqa_techqa_tar(
    split_name: str,
    max_chars_per_doc: int | None,
    max_documents: int | None,
    max_queries: int | None,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, str],
    dict[str, dict[str, int]],
    dict[str, list[dict[str, Any]]],
    dict[str, dict[str, Any]],
    bool,
]:
    split_norm = split_name.lower()
    if split_norm in {"test", "validation", "val"}:
        qa_member = "TechQA/validation/validation_reference.json"
        docs_member = "TechQA/validation/validation_technotes.json"
    elif split_norm in {"dev"}:
        qa_member = "TechQA/training_and_dev/dev_Q_A.json"
        docs_member = "TechQA/training_and_dev/training_dev_technotes.json"
    elif split_norm in {"train", "training"}:
        qa_member = "TechQA/training_and_dev/training_Q_A.json"
        docs_member = "TechQA/training_and_dev/training_dev_technotes.json"
    else:
        raise ValueError(
            f"Unsupported TechQA split '{split_name}'. Use one of: train, dev, test (test maps to validation files)."
        )

    tar_path = _resolve_primeqa_techqa_tar()
    with tarfile.open(tar_path, "r:gz") as tar:
        with tar.extractfile(docs_member) as f_docs:
            if f_docs is None:
                raise RuntimeError(f"Missing member in TechQA tar: {docs_member}")
            docs_raw = json.load(f_docs)

        with tar.extractfile(qa_member) as f_qa:
            if f_qa is None:
                raise RuntimeError(f"Missing member in TechQA tar: {qa_member}")
            qa_raw = json.load(f_qa)

    # docs file is a dict keyed by doc id.
    doc_items: list[tuple[str, dict[str, Any]]] = list(docs_raw.items()) if isinstance(docs_raw, dict) else []
    if max_documents is not None:
        doc_items = doc_items[: max_documents]
    allowed_doc_ids = {doc_id for doc_id, _ in doc_items}

    documents: dict[str, dict[str, Any]] = {}
    for doc_id, payload in doc_items:
        if not isinstance(payload, dict):
            continue
        title = _stringify(payload.get("title"))
        text = _stringify(payload.get("text") or payload.get("content"))
        text = _normalize_doc_text(text, max_chars_per_doc)
        if not text:
            continue
        documents[doc_id] = {"title": title or _safe_title(text), "text": text}

    qa_items = qa_raw if isinstance(qa_raw, list) else []
    if max_queries is not None:
        qa_items = qa_items[: max_queries]

    queries: dict[str, str] = {}
    qrels: dict[str, dict[str, int]] = {}
    evidences: dict[str, list[dict[str, Any]]] = {}
    answers: dict[str, dict[str, Any]] = {}

    for row_idx, row in enumerate(qa_items):
        if not isinstance(row, dict):
            continue

        query_id = _first_non_empty(row, ["QUESTION_ID", "question_id", "id"]) or f"techqa-q-{row_idx:06d}"
        question = _first_non_empty(row, ["QUESTION_TEXT", "question", "text", "QUESTION_TITLE"])
        if not question:
            continue

        queries[query_id] = question

        candidate_doc_ids = row.get("DOC_IDS")
        ranked_doc_ids: list[str] = []
        if isinstance(candidate_doc_ids, list):
            ranked_doc_ids = [_stringify(x) for x in candidate_doc_ids if _stringify(x)]

        # If document cap is used, keep only ids we actually loaded.
        if allowed_doc_ids:
            ranked_doc_ids = [d for d in ranked_doc_ids if d in allowed_doc_ids]
        ranked_doc_ids = [d for d in ranked_doc_ids if d in documents]

        if ranked_doc_ids:
            qrels[query_id] = {doc_id: 1 for doc_id in ranked_doc_ids}

        answerable_flag = _stringify(row.get("ANSWERABLE")).upper()
        gold_doc_id = _stringify(row.get("DOCUMENT"))
        if gold_doc_id in {"", "-"}:
            gold_doc_id = ""
        if gold_doc_id and gold_doc_id in documents:
            qrels.setdefault(query_id, {})
            qrels[query_id][gold_doc_id] = max(2, qrels[query_id].get(gold_doc_id, 0))

        answer_text = _stringify(row.get("ANSWER"))
        if answer_text == "-":
            answer_text = ""

        reference_answers = [answer_text] if answer_text else []
        is_answerable = answerable_flag == "Y" or bool(reference_answers)

        answers[query_id] = {
            "doc_id": gold_doc_id or (ranked_doc_ids[0] if ranked_doc_ids else ""),
            "reference_answers": reference_answers,
            "answerable": is_answerable,
        }

        evidence_entries: list[dict[str, Any]] = []
        ev_doc_id = answers[query_id]["doc_id"]
        if ev_doc_id and ev_doc_id in documents:
            evidence_text = documents[ev_doc_id]["text"]
            start_raw = _stringify(row.get("START_OFFSET"))
            end_raw = _stringify(row.get("END_OFFSET"))
            if start_raw.isdigit() and end_raw.isdigit():
                start = int(start_raw)
                end = int(end_raw)
                if 0 <= start < end <= len(evidence_text):
                    span = evidence_text[start:end].strip()
                    if span:
                        evidence_text = span
            evidence_entries = [{"doc_id": ev_doc_id, "evidence_text": evidence_text}]
        evidences[query_id] = evidence_entries or [{"doc_id": "", "evidence_text": ""}]

    if not qrels:
        raise RuntimeError("No qrels could be built from PrimeQA/TechQA source split.")

    allowed_qids = set(qrels.keys())
    queries = {qid: q for qid, q in queries.items() if qid in allowed_qids}
    answers = {qid: a for qid, a in answers.items() if qid in allowed_qids}
    evidences = {qid: e for qid, e in evidences.items() if qid in allowed_qids}

    return documents, queries, qrels, evidences, answers, True


def main() -> None:
    args = parse_args()

    target_dir = Path(args.target_dir)
    qrels_dir = target_dir / "qrels"
    target_dir.mkdir(parents=True, exist_ok=True)
    qrels_dir.mkdir(parents=True, exist_ok=True)

    three_config_mode = bool(args.docs_config and args.queries_config and args.qrels_config)

    if three_config_mode:
        print(
            f"[INFO] Loading docs from '{args.hf_dataset}' "
            f"config='{args.docs_config}' split='{args.source_split}'..."
        )
        docs_ds = _load_split(args.hf_dataset, args.source_split, args.docs_config)

        print(
            f"[INFO] Loading queries from '{args.hf_dataset}' "
            f"config='{args.queries_config}' split='{args.source_split}'..."
        )
        queries_ds = _load_split(args.hf_dataset, args.source_split, args.queries_config)

        print(
            f"[INFO] Loading qrels from '{args.hf_dataset}' "
            f"config='{args.qrels_config}' split='{args.source_split}'..."
        )
        qrels_ds = _load_split(args.hf_dataset, args.source_split, args.qrels_config)

        if args.max_documents is not None:
            docs_ds = docs_ds.select(range(min(args.max_documents, len(docs_ds))))
        if args.max_queries is not None:
            queries_ds = queries_ds.select(range(min(args.max_queries, len(queries_ds))))

        documents, queries, qrels, evidences, answers, source_has_answer_fields = _build_from_three_config_mode(
            docs_ds,
            queries_ds,
            qrels_ds,
            args.max_chars_per_doc,
        )

    else:
        if args.hf_dataset.lower() == "primeqa/techqa":
            print(
                "[INFO] Using PrimeQA/TechQA native TAR parser "
                f"for split='{args.source_split}'..."
            )
            documents, queries, qrels, evidences, answers, source_has_answer_fields = _build_from_primeqa_techqa_tar(
                split_name=args.source_split,
                max_chars_per_doc=args.max_chars_per_doc,
                max_documents=args.max_documents,
                max_queries=args.max_queries,
            )
        else:
            print(
                f"[INFO] Loading single QA dataset from '{args.hf_dataset}' split='{args.source_split}'..."
            )
            qa_ds = _load_split(args.hf_dataset, args.source_split)
            if args.max_queries is not None:
                qa_ds = qa_ds.select(range(min(args.max_queries, len(qa_ds))))

            documents, queries, qrels, evidences, answers, source_has_answer_fields = _build_from_single_dataset_mode(
                qa_ds,
                args.max_chars_per_doc,
            )

    if not documents:
        raise RuntimeError("No documents were created from source dataset.")
    if not queries:
        raise RuntimeError("No queries were created from source dataset.")
    if not qrels:
        raise RuntimeError("No qrels were created from source dataset.")

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

    non_empty_evidence_queries = sum(
        1
        for annotations in evidences.values()
        if any(_stringify(item.get("evidence_text")) for item in annotations if isinstance(item, dict))
    )
    non_empty_answer_queries = sum(
        1 for answer in answers.values() if isinstance(answer, dict) and answer.get("reference_answers")
    )

    metadata = {
        "dataset_name": "beir/techqa",
        "processor": "techqa_hf_converter",
        "split": args.output_split,
        "source": (
            f"huggingface:{args.hf_dataset}:"
            f"{args.docs_config},{args.queries_config},{args.qrels_config}:{args.source_split}"
            if three_config_mode
            else f"huggingface:{args.hf_dataset}:{args.source_split}"
        ),
        "num_documents": len(documents),
        "num_queries": len(queries),
        "num_qrels_queries": len(qrels),
        "normalized_schema_version": "1.0",
        "has_reference_answers_in_source": source_has_answer_fields,
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
    print(
        "[INFO] Evidence coverage: "
        f"queries_with_non_empty_evidence={non_empty_evidence_queries}/{len(queries)}"
    )
    print(
        "[INFO] Answer coverage: "
        + (
            f"queries_with_non_empty_reference_answers={non_empty_answer_queries}/{len(queries)}"
            if source_has_answer_fields
            else "N/A (source split has no answer annotation fields)"
        )
    )


if __name__ == "__main__":
    main()
