from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from src.embeddings.factory import EmbedderFactory
from src.retrieval.factory import RetrieverFactory

from .base import BaseExtrinsicEvaluator
from .common import build_base_row, build_run_context
from .io import (
    load_answers,
    load_index_metadata,
    load_queries,
    resolve_answers_path,
    resolve_queries_path,
)


class AnswerGenerationEvaluator(BaseExtrinsicEvaluator):
    _bertscorer_cache: dict[tuple[str, str, bool, str], Any] = {}
    PAPER_TOP_K_FOR_GENERATION = 5
    PAPER_BERTSCORE_MODEL = "microsoft/deberta-xlarge-mnli"
    GENERATION_MODEL_LABEL = "extractive-no-llm"

    @property
    def task_name(self) -> str:
        return "answer_generation"

    def evaluate(
        self,
        *,
        config: dict[str, Any],
        index_path: Path,
        index_metadata_path: Path,
        ks: Sequence[int],
    ) -> list[dict[str, Any]]:
        try:
            answers_file = resolve_answers_path(config)
        except Exception:
            answers_file = None

        if not ks:
            return self._build_rows(
                config=config,
                ks=[None],
                index_path=index_path,
                index_metadata_path=index_metadata_path,
                status="skipped",
                details="Task skipped: empty ks.",
                answers_path=None,
                generation_model=self.GENERATION_MODEL_LABEL,
            )

        resolved_ks = sorted({int(k) for k in ks})

        if answers_file is None:
            return self._build_rows(
                config=config,
                ks=resolved_ks,
                index_path=index_path,
                index_metadata_path=index_metadata_path,
                status="skipped",
                details="Task skipped: missing answers_path in YAML.",
                answers_path=None,
                generation_model=self.GENERATION_MODEL_LABEL,
            )

        if not answers_file.exists():
            return self._build_rows(
                config=config,
                ks=resolved_ks,
                index_path=index_path,
                index_metadata_path=index_metadata_path,
                status="skipped",
                details=f"Task skipped: answer annotations file not found at {answers_file}.",
                answers_path=str(answers_file),
                generation_model=self.GENERATION_MODEL_LABEL,
            )

        queries_path = resolve_queries_path(config)
        queries = load_queries(queries_path)
        answers = load_answers(answers_file)
        items = load_index_metadata(index_metadata_path)["items"]

        ordered_qids = [qid for qid in queries.keys() if qid in answers]
        if not ordered_qids:
            return self._build_rows(
                config=config,
                ks=resolved_ks,
                index_path=index_path,
                index_metadata_path=index_metadata_path,
                status="skipped",
                details="Task skipped: no overlapping query ids between queries and answers.",
                answers_path=str(answers_file),
                generation_model=self.GENERATION_MODEL_LABEL,
            )

        retrieval_cfg = config["retrieval"]
        embedding_cfg = config["embedding"]
        embedder = self._load_embedder(embedding_cfg)
        retriever = self._load_retriever(retrieval_cfg)
        retriever.load_index(str(index_path), str(index_metadata_path))

        query_texts = [queries[qid] for qid in ordered_qids]
        query_embeddings = self._encode_queries(embedder, query_texts, embedding_cfg)
        search_output = retriever.search(
            query_embeddings,
            top_k=self.PAPER_TOP_K_FOR_GENERATION,
        )
        all_indices = search_output["indices"]

        evaluable_qids = [
            qid
            for qid in ordered_qids
            if answers.get(qid, {}).get("answerable", False)
            and answers.get(qid, {}).get("reference_answers")
        ]
        if not evaluable_qids:
            return self._build_rows(
                config=config,
                ks=resolved_ks,
                index_path=index_path,
                index_metadata_path=index_metadata_path,
                status="skipped",
                details="Task skipped: no answerable queries with reference answers found.",
                answers_path=str(answers_file),
                generation_model=self.GENERATION_MODEL_LABEL,
            )

        row_idx_by_qid = {qid: idx for idx, qid in enumerate(ordered_qids)}
        refs_by_qid = {
            qid: [ref for ref in answers[qid]["reference_answers"] if ref.strip()]
            for qid in evaluable_qids
        }

        generated_answers: list[str] = []
        for qid in evaluable_qids:
            row_idx = row_idx_by_qid[qid]
            top_chunk_indices = all_indices[row_idx].tolist()[: self.PAPER_TOP_K_FOR_GENERATION]
            candidate_texts = self._collect_chunk_texts(top_chunk_indices, items)
            generated_answers.append(
                self._generate_answer_from_chunks(
                    query=queries[qid],
                    candidate_chunk_texts=candidate_texts,
                )
            )

        generated_embeddings, _ = embedder.encode_texts(generated_answers, embedding_cfg)
        generated_embeddings = np.asarray(generated_embeddings, dtype=np.float32)

        evaluable_query_embeddings, _ = embedder.encode_texts(
            [queries[qid] for qid in evaluable_qids],
            embedding_cfg,
        )
        evaluable_query_embeddings = np.asarray(evaluable_query_embeddings, dtype=np.float32)

        qa_scores: list[float] = []
        for idx in range(len(evaluable_qids)):
            qa_scores.append(
                self._cosine_similarity(
                    evaluable_query_embeddings[idx],
                    generated_embeddings[idx],
                )
            )

        bertscore_f1, bertscore_warning = self._compute_bertscore_macro_f1(
            generated_answers=generated_answers,
            references=[refs_by_qid[qid] for qid in evaluable_qids],
        )

        shared_metrics = {
            "qa_similarity": float(sum(qa_scores) / len(qa_scores)) if qa_scores else 0.0,
            "bertscore_f1": bertscore_f1,
        }
        metrics_by_k: dict[int, dict[str, float | None]] = {
            k: dict(shared_metrics) for k in resolved_ks
        }

        notes: list[str] = []
        notes.append("generation backend is extractive heuristic; no external LLM used")
        if any(k != self.PAPER_TOP_K_FOR_GENERATION for k in resolved_ks):
            notes.append(
                "paper uses top-5 chunks for answer generation; metrics replicated across requested ks"
            )
        notes.append("paper uses LLM-generated answers; this run uses extractive generation")
        if bertscore_warning:
            notes.append(f"BERTScore unavailable: {bertscore_warning}")

        details = "; ".join(notes) if notes else None

        return self._build_scored_rows(
            config=config,
            ks=resolved_ks,
            metrics_by_k=metrics_by_k,
            num_queries=len(ordered_qids),
            num_evaluable_queries=len(evaluable_qids),
            queries_path=str(queries_path),
            index_path=index_path,
            index_metadata_path=index_metadata_path,
            details=details,
            answers_path=str(answers_file),
            generation_model=self.GENERATION_MODEL_LABEL,
        )

    @staticmethod
    def _load_embedder(embedding_cfg: dict[str, Any]):
        embedder_name = embedding_cfg.get("name")
        if not isinstance(embedder_name, str) or not embedder_name.strip():
            raise ValueError("Embedding config requires a non-empty 'name' for extrinsic evaluation.")
        return EmbedderFactory.create(embedder_name.strip())

    @staticmethod
    def _load_retriever(retrieval_cfg: dict[str, Any]):
        retriever_name = retrieval_cfg.get("name", "")
        return RetrieverFactory.create(str(retriever_name).strip().lower())

    @staticmethod
    def _encode_queries(embedder, query_texts: list[str], embedding_cfg: dict[str, Any]):
        query_embeddings, _ = embedder.encode_texts(query_texts, embedding_cfg)
        return query_embeddings

    @staticmethod
    def _collect_chunk_texts(chunk_indices: list[int], items: list[dict[str, Any]]) -> list[str]:
        texts: list[str] = []
        for chunk_idx in chunk_indices:
            if chunk_idx < 0 or chunk_idx >= len(items):
                continue
            text = items[chunk_idx].get("text", "")
            if isinstance(text, str) and text.strip():
                texts.append(text.strip())
        return texts

    @staticmethod
    def _normalize_tokens(text: str) -> list[str]:
        text = text.lower()
        text = re.sub(r"\b(a|an|the)\b", " ", text)
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text.split() if text else []

    @classmethod
    def _sentence_score(cls, sentence: str, query_tokens: set[str]) -> float:
        sentence_tokens = set(cls._normalize_tokens(sentence))
        if not sentence_tokens:
            return 0.0
        overlap = len(sentence_tokens & query_tokens)
        coverage = overlap / float(len(query_tokens)) if query_tokens else 0.0
        density = overlap / float(len(sentence_tokens))
        return 0.7 * coverage + 0.3 * density

    @classmethod
    def _generate_answer_from_chunks(
        cls,
        *,
        query: str,
        candidate_chunk_texts: list[str],
    ) -> str:
        if not candidate_chunk_texts:
            return ""
        query_tokens = set(cls._normalize_tokens(query))
        candidate_sentences: list[str] = []
        for chunk_text in candidate_chunk_texts:
            pieces = re.split(r"(?<=[.!?])\s+|\n+", chunk_text)
            for piece in pieces:
                sentence = piece.strip()
                if sentence:
                    candidate_sentences.append(sentence)

        if not candidate_sentences:
            return candidate_chunk_texts[0][:512]

        scored = [
            (cls._sentence_score(sentence, query_tokens), idx, sentence)
            for idx, sentence in enumerate(candidate_sentences)
        ]
        scored.sort(key=lambda x: (-x[0], x[1]))

        selected = [text for _, _, text in scored[:3]]
        generated = " ".join(selected).strip()
        if not generated:
            generated = candidate_chunk_texts[0]
        return generated[:768]

    @staticmethod
    def _cosine_similarity(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
        a = np.asarray(emb_a, dtype=np.float32)
        b = np.asarray(emb_b, dtype=np.float32)
        a_norm = float(np.linalg.norm(a))
        b_norm = float(np.linalg.norm(b))
        if a_norm <= 0.0 or b_norm <= 0.0:
            return 0.0
        return float(np.dot(a, b) / (a_norm * b_norm))

    @classmethod
    def _compute_bertscore_macro_f1(
        cls,
        *,
        generated_answers: list[str],
        references: list[list[str]],
    ) -> tuple[float | None, str | None]:
        try:
            from bert_score import BERTScorer
        except Exception as e:
            return None, str(e)

        flat_cands: list[str] = []
        flat_refs: list[str] = []
        offsets: list[tuple[int, int]] = []

        for candidate, refs in zip(generated_answers, references):
            start = len(flat_cands)
            if refs:
                for ref in refs:
                    flat_cands.append(candidate)
                    flat_refs.append(ref)
            end = len(flat_cands)
            offsets.append((start, end))

        if not flat_cands:
            return 0.0, None

        device = cls._resolve_bertscore_device()
        try:
            scorer = cls._get_bertscorer(
                BERTScorer=BERTScorer,
                model_type=cls.PAPER_BERTSCORE_MODEL,
                lang="en",
                rescale_with_baseline=True,
                device=device,
            )
            _, _, f1 = scorer.score(flat_cands, flat_refs, verbose=False)

            per_query_best: list[float] = []
            for start, end in offsets:
                if start == end:
                    per_query_best.append(0.0)
                    continue
                per_query_best.append(float(torch_max(f1[start:end])))

            if not per_query_best:
                return 0.0, None
            score_value = float(sum(per_query_best) / len(per_query_best))
            return score_value, None
        except Exception as e:
            return None, f"{cls.PAPER_BERTSCORE_MODEL}: {e}"

    @classmethod
    def _resolve_bertscore_device(cls) -> str:
        try:
            import torch
        except Exception:
            return "cpu"
        # BERTScore on large DeBERTa checkpoints can be unstable on MPS in some environments.
        # Keep it on CPU for deterministic behavior.
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    @classmethod
    def _get_bertscorer(
        cls,
        *,
        BERTScorer: Any,
        model_type: str,
        lang: str,
        rescale_with_baseline: bool,
        device: str,
    ) -> Any:
        key = (model_type, lang, rescale_with_baseline, device)
        cached = cls._bertscorer_cache.get(key)
        if cached is not None:
            return cached

        scorer = BERTScorer(
            model_type=model_type,
            lang=lang,
            rescale_with_baseline=rescale_with_baseline,
            device=device,
        )
        cls._bertscorer_cache[key] = scorer
        return scorer

    def _build_scored_rows(
        self,
        *,
        config: dict[str, Any],
        ks: Sequence[int | None],
        metrics_by_k: dict[int, dict[str, float | None]],
        num_queries: int,
        num_evaluable_queries: int,
        queries_path: str,
        index_path: Path,
        index_metadata_path: Path,
        details: str | None,
        answers_path: str | None,
        generation_model: str | None,
    ) -> list[dict[str, Any]]:
        context = build_run_context(
            config=config,
            task_name=self.task_name,
            default_experiment_name="answer-generation",
        )
        rows: list[dict[str, Any]] = []
        for k in ks:
            if k is None:
                continue
            row = build_base_row(
                context=context,
                k=k,
                index_path=index_path,
                index_metadata_path=index_metadata_path,
            )
            row.update(
                {
                    "qa_similarity": metrics_by_k[k]["qa_similarity"],
                    "bertscore_f1": metrics_by_k[k]["bertscore_f1"],
                    "num_queries": num_queries,
                    "num_evaluable_queries": num_evaluable_queries,
                    "queries_path": queries_path,
                    "answers_path": answers_path,
                    "generation_model": generation_model,
                }
            )
            if details:
                row["status"] = "partial"
                row["details"] = details
            rows.append(row)
        return rows

    def _build_rows(
        self,
        *,
        config: dict[str, Any],
        ks: Sequence[int | None],
        index_path: Path,
        index_metadata_path: Path,
        status: str,
        details: str,
        answers_path: str | None,
        generation_model: str | None,
    ) -> list[dict[str, Any]]:
        context = build_run_context(
            config=config,
            task_name=self.task_name,
            default_experiment_name="answer-generation",
        )
        rows: list[dict[str, Any]] = []
        for k in ks:
            row = build_base_row(
                context=context,
                k=k,
                index_path=index_path,
                index_metadata_path=index_metadata_path,
            )
            row.update(
                {
                    "status": status,
                    "details": details,
                    "answers_path": answers_path,
                    "generation_model": generation_model,
                }
            )
            rows.append(row)
        return rows


def torch_max(values: Any) -> float:
    if hasattr(values, "max"):
        max_value = values.max()
        if hasattr(max_value, "item"):
            return float(max_value.item())
        return float(max_value)
    return float(max(values))
