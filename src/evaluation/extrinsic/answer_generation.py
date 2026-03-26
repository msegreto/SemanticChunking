from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from src.embeddings.factory import EmbedderFactory

from .common import build_base_row, build_run_context
from .io import (
    get_extrinsic_task_cfg,
    load_answers,
    load_items_by_docno,
    load_queries,
    load_run_tsv,
    resolve_answers_path,
    resolve_queries_path,
)


class AnswerGenerationEvaluator:
    _bertscorer_cache: dict[tuple[str, str, bool, str], Any] = {}
    PAPER_TOP_K_FOR_GENERATION = 5
    PAPER_BERTSCORE_MODEL = "microsoft/deberta-xlarge-mnli"

    @property
    def task_name(self) -> str:
        return "answer_generation"

    def evaluate(
        self,
        *,
        config: dict[str, Any],
        retrieval_output: dict[str, Any],
        ks: Sequence[int],
    ) -> list[dict[str, Any]]:
        manifest_path = Path(str(retrieval_output.get("manifest_path", "")))
        run_path = Path(str(retrieval_output.get("run_path", "")))
        items_path = Path(str(retrieval_output.get("items_path", "")))
        try:
            answers_file = resolve_answers_path(config)
        except Exception:
            answers_file = None

        if not ks:
            return self._build_rows(
                config=config,
                ks=[None],
                manifest_path=manifest_path,
                run_path=run_path,
                items_path=items_path,
                status="skipped",
                details="Task skipped: empty ks.",
                answers_path=None,
                generation_model=None,
            )

        resolved_ks = sorted({int(k) for k in ks})

        if answers_file is None:
            return self._build_rows(
                config=config,
                ks=resolved_ks,
                manifest_path=manifest_path,
                run_path=run_path,
                items_path=items_path,
                status="skipped",
                details="Task skipped: missing answers_path in YAML.",
                answers_path=None,
                generation_model=None,
            )

        if not answers_file.exists():
            return self._build_rows(
                config=config,
                ks=resolved_ks,
                manifest_path=manifest_path,
                run_path=run_path,
                items_path=items_path,
                status="skipped",
                details=f"Task skipped: answer annotations file not found at {answers_file}.",
                answers_path=str(answers_file),
                generation_model=None,
            )

        queries_path = resolve_queries_path(config)
        queries = load_queries(queries_path)
        answers = load_answers(answers_file)
        items_by_docno = load_items_by_docno(items_path)
        run_by_qid = load_run_tsv(run_path)

        ordered_qids = [qid for qid in queries.keys() if qid in answers]
        if not ordered_qids:
            return self._build_rows(
                config=config,
                ks=resolved_ks,
                manifest_path=manifest_path,
                run_path=run_path,
                items_path=items_path,
                status="skipped",
                details="Task skipped: no overlapping query ids between queries and answers.",
                answers_path=str(answers_file),
                generation_model=None,
            )

        generation_cfg = self._resolve_generation_config(config)
        embedding_cfg = config["embedding"]
        embedder = self._load_embedder(embedding_cfg)

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
                manifest_path=manifest_path,
                run_path=run_path,
                items_path=items_path,
                status="skipped",
                details="Task skipped: no answerable queries with reference answers found.",
                answers_path=str(answers_file),
                generation_model=None,
            )

        refs_by_qid = {
            qid: [ref for ref in answers[qid]["reference_answers"] if ref.strip()]
            for qid in evaluable_qids
        }

        generated_answers, generation_notes, generation_model = self._generate_answers_with_pyterrier_rag(
            generation_cfg=generation_cfg,
            evaluable_qids=evaluable_qids,
            queries=queries,
            run_by_qid=run_by_qid,
            items_by_docno=items_by_docno,
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
        if generation_notes:
            notes.extend(generation_notes)
        if any(k != generation_cfg["top_k"] for k in resolved_ks):
            notes.append(
                f"generation used top-{generation_cfg['top_k']} chunks; metrics replicated across requested ks"
            )
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
            manifest_path=manifest_path,
            run_path=run_path,
            items_path=items_path,
            details=details,
            answers_path=str(answers_file),
            generation_model=generation_model,
        )

    @staticmethod
    def _load_embedder(embedding_cfg: dict[str, Any]):
        embedder_name = embedding_cfg.get("name")
        if not isinstance(embedder_name, str) or not embedder_name.strip():
            raise ValueError("Embedding config requires a non-empty 'name' for extrinsic evaluation.")
        return EmbedderFactory.create(embedder_name.strip())

    @staticmethod
    def _resolve_generation_config(config: dict[str, Any]) -> dict[str, Any]:
        task_cfg = get_extrinsic_task_cfg(config, "answer_generation")
        top_k = int(task_cfg.get("top_k", AnswerGenerationEvaluator.PAPER_TOP_K_FOR_GENERATION))
        if top_k <= 0:
            raise ValueError("'evaluation.extrinsic_tasks.answer_generation.top_k' must be a positive integer.")
        reader_model = str(task_cfg.get("reader_model", "google/flan-t5-base")).strip()
        reader_backend = str(task_cfg.get("reader_backend", "seq2seqlm")).strip().lower().replace("-", "_")
        return {
            "top_k": top_k,
            "reader_model": reader_model,
            "reader_backend": reader_backend,
        }

    def _generate_answers_with_pyterrier_rag(
        self,
        *,
        generation_cfg: dict[str, Any],
        evaluable_qids: list[str],
        queries: dict[str, str],
        run_by_qid: dict[str, list[dict[str, Any]]],
        items_by_docno: dict[str, dict[str, Any]],
    ) -> tuple[list[str], list[str], str]:
        from pyterrier_rag import Seq2SeqLMBackend
        from pyterrier_rag.readers import Reader

        if generation_cfg["reader_backend"] != "seq2seqlm":
            raise ValueError(
                "Only reader_backend=seq2seqlm is currently supported for pyterrier_rag integration."
            )

        reader = Reader(Seq2SeqLMBackend(generation_cfg["reader_model"]))
        reader_input = self._build_pyterrier_rag_input(
            top_k=int(generation_cfg["top_k"]),
            evaluable_qids=evaluable_qids,
            queries=queries,
            run_by_qid=run_by_qid,
            items_by_docno=items_by_docno,
        )

        if reader_input.empty:
            return ["" for _ in evaluable_qids], [
                "pyterrier_rag input is empty; no chunks available for generation",
            ], generation_cfg["reader_model"]

        reader_output = reader.transform(reader_input)
        answers_by_qid = self._extract_answers_from_reader_output(reader_output)

        generated_answers: list[str] = []
        missing_answers = 0
        for qid in evaluable_qids:
            answer = answers_by_qid.get(qid, "").strip()
            if not answer:
                missing_answers += 1
            generated_answers.append(answer)

        notes = [
            f"generation backend is pyterrier_rag reader ({generation_cfg['reader_model']})",
        ]
        if missing_answers > 0:
            notes.append(
                f"pyterrier_rag returned no answer for {missing_answers} queries"
            )
        return generated_answers, notes, generation_cfg["reader_model"]

    @staticmethod
    def _build_pyterrier_rag_input(
        *,
        top_k: int,
        evaluable_qids: list[str],
        queries: dict[str, str],
        run_by_qid: dict[str, list[dict[str, Any]]],
        items_by_docno: dict[str, dict[str, Any]],
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for qid in evaluable_qids:
            query = queries[qid]
            for rank, hit in enumerate(run_by_qid.get(qid, [])[:top_k]):
                docno = str(hit.get("docno") or "")
                if not docno:
                    continue
                item = items_by_docno.get(docno)
                if item is None:
                    continue
                text = str(item.get("text", "")).strip()
                if not text:
                    continue
                rows.append(
                    {
                        "qid": qid,
                        "query": query,
                        "docno": docno,
                        "text": text,
                        "title": str(item.get("title", "") or ""),
                        "score": float(hit.get("score", 0.0)),
                        "rank": rank,
                    }
                )
        return pd.DataFrame(rows)

    @staticmethod
    def _extract_answers_from_reader_output(reader_output: Any) -> dict[str, str]:
        if not isinstance(reader_output, pd.DataFrame) or reader_output.empty:
            return {}

        answer_column = None
        for candidate in ("answer", "response", "generated_text", "generation", "output", "text"):
            if candidate in reader_output.columns:
                answer_column = candidate
                break
        if answer_column is None or "qid" not in reader_output.columns:
            return {}

        answers_by_qid: dict[str, str] = {}
        grouped = reader_output.groupby("qid", sort=False)
        for qid, group in grouped:
            answer = str(group.iloc[0].get(answer_column, "") or "").strip()
            if answer:
                answers_by_qid[str(qid)] = answer
        return answers_by_qid

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
        manifest_path: Path,
        run_path: Path,
        items_path: Path,
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
                retrieval_manifest_path=manifest_path,
                retrieval_run_path=run_path,
                retrieval_items_path=items_path,
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
        manifest_path: Path,
        run_path: Path,
        items_path: Path,
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
                retrieval_manifest_path=manifest_path,
                retrieval_run_path=run_path,
                retrieval_items_path=items_path,
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
