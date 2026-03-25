from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config.loader import load_experiment_config


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class StateEntry:
    key: str
    config_path: Path
    attempts: int
    status: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rerun document_retrieval extrinsic evaluation for experiments listed in a "
            "run_experiments_resume state file."
        )
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        required=True,
        help="State JSON file (e.g. results/runner_state2/grid_search_stage1_bge.json).",
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("logs/rerun_document_retrieval"),
        help="Directory for per-experiment rerun logs.",
    )
    parser.add_argument(
        "--history-logs-dir",
        type=Path,
        default=Path("logs"),
        help=(
            "Directory containing original run logs used to infer 'already run' experiments "
            "when state attempts/status are stale (default: logs)."
        ),
    )
    parser.add_argument(
        "--statuses",
        type=str,
        default="completed,failed,interrupted,running",
        help=(
            "Comma-separated statuses to include from state file "
            "(default: completed,failed,interrupted,running)."
        ),
    )
    parser.add_argument(
        "--include-attempts-zero",
        action="store_true",
        help="Also include entries with attempts=0.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of experiments to run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected runs and commands without executing.",
    )
    parser.add_argument(
        "--python-executable",
        type=Path,
        default=None,
        help=(
            "Python executable used to run each command. "
            "Default: auto-detect .venv/bin/python, otherwise current interpreter."
        ),
    )
    return parser.parse_args()


def _resolve_python_executable(override: Path | None) -> str:
    if override is not None:
        return str(override)
    venv_python = REPO_ROOT / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"State file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid state format (expected JSON object): {path}")
    return payload


def _resolve_selection(
    state_payload: dict[str, Any],
    *,
    allowed_statuses: set[str],
    include_attempts_zero: bool,
    history_logs_dir: Path | None,
    limit: int | None,
) -> list[StateEntry]:
    experiments = state_payload.get("experiments", {})
    if not isinstance(experiments, dict):
        raise ValueError("Invalid state file: 'experiments' must be a JSON object.")

    selected: list[StateEntry] = []
    for key in sorted(experiments.keys()):
        item = experiments.get(key, {})
        if not isinstance(item, dict):
            continue

        raw_path = item.get("config_path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            continue

        attempts = int(item.get("attempts", 0) or 0)
        status = str(item.get("status", "pending")).strip().lower()

        inferred_from_log = False
        if history_logs_dir is not None:
            inferred_from_log = (history_logs_dir / f"{key}.txt").exists()
        if status not in allowed_statuses and not inferred_from_log:
            continue

        already_ran = attempts > 0 or inferred_from_log
        if not include_attempts_zero and not already_ran:
            continue

        selected.append(
            StateEntry(
                key=key,
                config_path=Path(raw_path),
                attempts=attempts,
                status=status,
            )
        )

    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be > 0 when provided.")
        selected = selected[:limit]
    return selected


def _resolve_index_dir(config: dict[str, Any], config_path: Path) -> Path:
    retrieval_cfg = config.get("retrieval", {})
    configured_output_dir = retrieval_cfg.get("output_dir")
    if isinstance(configured_output_dir, str) and configured_output_dir.strip():
        return Path(configured_output_dir)

    dataset_name = str(config.get("dataset", {}).get("name", "unknown-dataset"))
    chunking_type = str(config.get("chunking", {}).get("type", "unknown-chunking"))
    experiment_id = config_path.stem
    return Path(f"data/indexes/{dataset_name}/{chunking_type}/{experiment_id}")


def _resolve_index_path(index_dir: Path) -> Path:
    manifest_path = index_dir / "manifest.json"
    fallback_candidates = [
        index_dir / "index.faiss",
        index_dir / "vectors.npy",
    ]

    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as e:
            raise RuntimeError(f"Unreadable manifest '{manifest_path}': {e}") from e
        raw_index_path = manifest.get("index_path")
        if isinstance(raw_index_path, str) and raw_index_path.strip():
            from_manifest = Path(raw_index_path.strip())
            if from_manifest.is_absolute():
                candidates = [from_manifest, index_dir / from_manifest.name, *fallback_candidates]
            else:
                candidates = [index_dir / from_manifest, *fallback_candidates]
        else:
            candidates = fallback_candidates
    else:
        candidates = fallback_candidates

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved

    raise FileNotFoundError(
        f"No index file found under '{index_dir}'. Checked manifest + fallbacks index.faiss/vectors.npy."
    )


def _write_doc_retrieval_temp_config(config: dict[str, Any], target_dir: Path, key: str) -> Path:
    patched = dict(config)
    evaluation = dict(patched.get("evaluation", {}))
    evaluation["extrinsic"] = True
    evaluation["extrinsic_evaluator"] = "document_retrieval"
    evaluation["extrinsic_tasks_to_run"] = ["document_retrieval"]
    patched["evaluation"] = evaluation

    target_dir.mkdir(parents=True, exist_ok=True)
    out_path = target_dir / f"{key}.yaml"
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(patched, f, sort_keys=False, allow_unicode=False)
    return out_path


def _run_one(
    entry: StateEntry,
    *,
    python_executable: str,
    logs_dir: Path,
    temp_config_dir: Path,
    dry_run: bool,
) -> tuple[bool, str]:
    if not entry.config_path.exists():
        return False, f"config not found: {entry.config_path}"

    try:
        config = load_experiment_config(entry.config_path)
    except Exception as e:
        return False, f"failed loading config: {e}"

    index_dir = _resolve_index_dir(config, entry.config_path)
    metadata_path = (index_dir / "metadata.pkl").resolve()
    manifest_path = (index_dir / "manifest.json").resolve()
    if not metadata_path.exists():
        return False, f"metadata missing: {metadata_path}"

    try:
        index_path = _resolve_index_path(index_dir)
    except Exception as e:
        return False, str(e)

    temp_config_path = _write_doc_retrieval_temp_config(config, temp_config_dir, entry.key)

    cmd = [
        python_executable,
        "-m",
        "scripts.run_extrinsic_eval",
        "--config",
        str(temp_config_path),
        "--index-path",
        str(index_path),
        "--index-metadata-path",
        str(metadata_path),
    ]
    if manifest_path.exists():
        cmd.extend(["--index-manifest-path", str(manifest_path)])

    if dry_run:
        return True, f"DRY-RUN {' '.join(cmd)}"

    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{entry.key}.txt"
    started_iso = _utc_now_iso()

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    root_str = str(REPO_ROOT)
    env["PYTHONPATH"] = f"{root_str}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else root_str
    env["CUDA_VISIBLE_DEVICES"] = "0"

    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"\n[{started_iso}] START {' '.join(cmd)}\n")
        log_file.flush()
        process = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        exit_code = process.wait()

    finished_iso = _utc_now_iso()
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"[{finished_iso}] END exit_code={exit_code}\n")

    if exit_code != 0:
        return False, f"run_extrinsic_eval failed (exit_code={exit_code})"
    return True, f"ok (log: {log_path})"


def main() -> None:
    args = parse_args()
    state_payload = _load_state(args.state_file)

    allowed_statuses = {
        token.strip().lower()
        for token in args.statuses.split(",")
        if token.strip()
    }
    if not allowed_statuses:
        raise ValueError("No valid statuses provided in --statuses.")

    selected = _resolve_selection(
        state_payload,
        allowed_statuses=allowed_statuses,
        include_attempts_zero=bool(args.include_attempts_zero),
        history_logs_dir=args.history_logs_dir,
        limit=args.limit,
    )

    print(f"[INFO] Selected experiments: {len(selected)}")
    if not selected:
        return

    python_executable = _resolve_python_executable(args.python_executable)
    print(f"[INFO] Python executable: {python_executable}")
    print(f"[INFO] Dry-run: {bool(args.dry_run)}")
    print(f"[INFO] Logs dir: {args.logs_dir.resolve()}")

    success_count = 0
    fail_count = 0
    failures: list[tuple[str, str]] = []

    with tempfile.TemporaryDirectory(prefix="doc_retrieval_rerun_") as tmp_dir_raw:
        temp_config_dir = Path(tmp_dir_raw)
        for idx, entry in enumerate(selected, start=1):
            print(
                f"[RUN {idx}/{len(selected)}] key={entry.key} status={entry.status} "
                f"attempts={entry.attempts}"
            )
            ok, message = _run_one(
                entry,
                python_executable=python_executable,
                logs_dir=args.logs_dir,
                temp_config_dir=temp_config_dir,
                dry_run=bool(args.dry_run),
            )
            if ok:
                success_count += 1
                print(f"[OK] {entry.key}: {message}")
            else:
                fail_count += 1
                failures.append((entry.key, message))
                print(f"[FAIL] {entry.key}: {message}")

    print("\n=== Summary ===")
    print(f"selected={len(selected)} success={success_count} failed={fail_count}")
    if failures:
        for key, reason in failures:
            print(f"- {key}: {reason}")


if __name__ == "__main__":
    main()
