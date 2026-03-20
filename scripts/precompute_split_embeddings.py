from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.embeddings.factory import EmbedderFactory


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")
    tmp.replace(path)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _slugify(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", text).strip("_").lower() or "default"


def _resolve_text(unit: dict[str, Any]) -> str:
    value = unit.get("text")
    if not isinstance(value, str):
        return ""
    return value.strip()


def _has_valid_cached_embedding(unit: dict[str, Any], cache_key: str, expected_text: str) -> bool:
    embeddings = unit.get("embeddings")
    if not isinstance(embeddings, dict):
        return False
    entry = embeddings.get(cache_key)
    if not isinstance(entry, dict):
        return False
    if entry.get("text") != expected_text:
        return False
    vector = entry.get("vector")
    return isinstance(vector, list) and len(vector) > 0


@dataclass
class Job:
    split_path: Path
    state_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Precompute sentence embeddings for split JSONL files and save them inside "
            "each row under embeddings.<cache_key>. This enables semantic chunkers to "
            "reuse cached vectors instead of recomputing."
        )
    )
    parser.add_argument(
        "--split-jsonl",
        type=Path,
        action="append",
        default=[],
        help="Path to a split JSONL file (can be repeated).",
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("data/processed"),
        help="Root used to discover split JSONL files when --split-jsonl is not passed.",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="**/sentences.jsonl",
        help="Glob used for discovery under --processed-root.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mpnet",
        help="Embedder name for EmbedderFactory (e.g. mpnet, bge, stella).",
    )
    parser.add_argument(
        "--cache-key",
        type=str,
        default=None,
        help=(
            "Key used in row.embeddings.<cache_key>. "
            "Default: same value as --model."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size used by the embedder.",
    )
    parser.add_argument(
        "--line-chunk-size",
        type=int,
        default=2000,
        help="Max JSONL rows buffered in memory before flushing to output.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute and overwrite cached embeddings even when already present.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with next files if one file fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print discovered jobs without changing files.",
    )
    return parser.parse_args()


def _discover_jobs(
    *,
    explicit_paths: list[Path],
    processed_root: Path,
    glob_expr: str,
    state_filename: str,
) -> list[Job]:
    jobs: list[Job] = []
    seen: set[Path] = set()

    if explicit_paths:
        for path in explicit_paths:
            resolved = path.expanduser().resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            jobs.append(Job(split_path=resolved, state_path=resolved.parent / state_filename))
        return jobs

    for path in sorted(processed_root.glob(glob_expr)):
        if not path.is_file():
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        jobs.append(Job(split_path=resolved, state_path=resolved.parent / state_filename))
    return jobs


def _embed_missing_texts(
    *,
    embedder: Any,
    missing_texts: list[str],
    batch_size: int,
) -> np.ndarray:
    vectors, _ = embedder.encode_texts(
        missing_texts,
        {
            "batch_size": batch_size,
            "normalize_embeddings": True,
            "show_progress_bar": False,
            "convert_to_numpy": True,
            "log_embedding_calls": False,
        },
    )
    return np.asarray(vectors, dtype=float)


def _flush_buffer(
    *,
    out_file: Any,
    embedder: Any,
    buffer: list[dict[str, Any]],
    missing_indices: list[int],
    missing_texts: list[str],
    cache_key: str,
    batch_size: int,
) -> tuple[int, int]:
    added_count = 0
    written_count = len(buffer)
    if missing_indices:
        vectors = _embed_missing_texts(
            embedder=embedder,
            missing_texts=missing_texts,
            batch_size=batch_size,
        )
        if vectors.shape[0] != len(missing_indices):
            raise ValueError("Embedding output rows do not match buffered missing rows.")

        for local_idx, vector in zip(missing_indices, vectors):
            row = buffer[local_idx]
            text = _resolve_text(row)
            row.setdefault("embeddings", {})
            row["embeddings"][cache_key] = {
                "text": text,
                "vector": vector.tolist(),
            }
            added_count += 1

    for row in buffer:
        out_file.write(json.dumps(row, ensure_ascii=False) + "\n")

    return written_count, added_count


def _process_one(
    *,
    job: Job,
    model_name: str,
    cache_key: str,
    batch_size: int,
    line_chunk_size: int,
    force: bool,
) -> tuple[int, int]:
    if not job.split_path.exists():
        raise FileNotFoundError(f"Split JSONL not found: {job.split_path}")

    old_state = _load_json(job.state_path)
    started_iso = _utc_now_iso()
    running_state = {
        "version": 1,
        "status": "running",
        "completed": False,
        "created_at": old_state.get("created_at", started_iso),
        "updated_at": started_iso,
        "started_at": started_iso,
        "finished_at": None,
        "duration_seconds": None,
        "split_path": str(job.split_path),
        "embedder_model": model_name,
        "cache_key": cache_key,
        "batch_size": batch_size,
        "line_chunk_size": line_chunk_size,
        "force": bool(force),
        "rows_read": 0,
        "rows_written": 0,
        "rows_added": 0,
        "error": None,
    }
    _save_json(job.state_path, running_state)

    tmp_path = job.split_path.with_suffix(job.split_path.suffix + f".{cache_key}.tmp")
    rows_read = 0
    rows_written = 0
    rows_added = 0
    flush_count = 0

    try:
        embedder = EmbedderFactory.create(model_name)
        with job.split_path.open("r", encoding="utf-8") as f_in, tmp_path.open("w", encoding="utf-8") as f_out:
            buffer: list[dict[str, Any]] = []
            missing_indices: list[int] = []
            missing_texts: list[str] = []

            for raw_line in f_in:
                line = raw_line.strip()
                if not line:
                    continue

                row = json.loads(line)
                text = _resolve_text(row)
                if not text:
                    raise ValueError("Found split row with empty/non-string text.")

                needs_embedding = force or not _has_valid_cached_embedding(row, cache_key, text)
                if needs_embedding:
                    missing_indices.append(len(buffer))
                    missing_texts.append(text)

                buffer.append(row)
                rows_read += 1

                if len(buffer) >= line_chunk_size:
                    written, added = _flush_buffer(
                        out_file=f_out,
                        embedder=embedder,
                        buffer=buffer,
                        missing_indices=missing_indices,
                        missing_texts=missing_texts,
                        cache_key=cache_key,
                        batch_size=batch_size,
                    )
                    rows_written += written
                    rows_added += added
                    flush_count += 1
                    buffer = []
                    missing_indices = []
                    missing_texts = []

                    if flush_count % 20 == 0:
                        print(
                            f"[INFO] {job.split_path.name} progress "
                            f"rows_read={rows_read} rows_added={rows_added}"
                        )

            if buffer:
                written, added = _flush_buffer(
                    out_file=f_out,
                    embedder=embedder,
                    buffer=buffer,
                    missing_indices=missing_indices,
                    missing_texts=missing_texts,
                    cache_key=cache_key,
                    batch_size=batch_size,
                )
                rows_written += written
                rows_added += added

        tmp_path.replace(job.split_path)

        finished = datetime.now(timezone.utc)
        finished_iso = finished.replace(microsecond=0).isoformat()
        duration = (finished - datetime.fromisoformat(started_iso)).total_seconds()
        final_state = dict(running_state)
        final_state.update(
            {
                "status": "completed",
                "completed": True,
                "updated_at": finished_iso,
                "finished_at": finished_iso,
                "duration_seconds": duration,
                "rows_read": rows_read,
                "rows_written": rows_written,
                "rows_added": rows_added,
                "error": None,
            }
        )
        _save_json(job.state_path, final_state)
        return rows_read, rows_added
    except Exception as e:
        failed_at = _utc_now_iso()
        fail_state = dict(running_state)
        fail_state.update(
            {
                "status": "failed",
                "completed": False,
                "updated_at": failed_at,
                "finished_at": failed_at,
                "rows_read": rows_read,
                "rows_written": rows_written,
                "rows_added": rows_added,
                "error": str(e),
            }
        )
        _save_json(job.state_path, fail_state)
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.line_chunk_size <= 0:
        raise ValueError("--line-chunk-size must be > 0")

    cache_key = (args.cache_key or args.model).strip()
    if not cache_key:
        raise ValueError("cache key resolved to empty string.")

    state_filename = f"state_embeddings_{_slugify(cache_key)}.json"
    jobs = _discover_jobs(
        explicit_paths=args.split_jsonl,
        processed_root=args.processed_root,
        glob_expr=args.glob,
        state_filename=state_filename,
    )
    if not jobs:
        print("[WARN] No split JSONL files found.")
        return

    print(f"[INFO] Found {len(jobs)} split file(s).")
    for job in jobs:
        print(f"[INFO] job split={job.split_path} state={job.state_path}")

    if args.dry_run:
        print("[DRY-RUN] No files changed.")
        return

    failures = 0
    for idx, job in enumerate(jobs, start=1):
        print(f"[INFO] ({idx}/{len(jobs)}) Embedding {job.split_path}")
        try:
            rows_read, rows_added = _process_one(
                job=job,
                model_name=args.model,
                cache_key=cache_key,
                batch_size=args.batch_size,
                line_chunk_size=args.line_chunk_size,
                force=args.force,
            )
            print(f"[OK] {job.split_path.name}: rows={rows_read}, newly_embedded={rows_added}")
        except Exception as e:
            failures += 1
            print(f"[FAIL] {job.split_path}: {e}")
            if not args.continue_on_error:
                break

    if failures > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
