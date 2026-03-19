from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class ExperimentEntry:
    key: str
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run all experiment YAML configs with persistent progress tracking. "
            "If interrupted, rerun to continue from missing/failed configs."
        )
    )
    parser.add_argument(
        "--configs-dir",
        type=Path,
        default=Path("configs/experiments/grid_search_stage1_stella"),
        help="Directory containing experiment YAML files (default: grid_search_stage1_stella).",
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        default=Path("results/runner_state/grid_search_stage1_stella.json"),
        help="State JSON path used to track progress.",
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("logs"),
        help="Directory where per-experiment logs are stored.",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="*.yaml",
        help="File glob to select configs inside --configs-dir.",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry experiments marked as failed in state file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be executed without running commands.",
    )
    parser.add_argument(
        "--python-executable",
        type=Path,
        default=None,
        help=(
            "Python executable used to run each experiment. "
            "Default: auto-detect .venv/bin/python, otherwise current interpreter."
        ),
    )
    return parser.parse_args()


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "created_at": _utc_now_iso(), "experiments": {}}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid state file format: {path}")
    data.setdefault("version", 1)
    data.setdefault("created_at", _utc_now_iso())
    data.setdefault("experiments", {})
    return data


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)
        f.write("\n")
    tmp_path.replace(path)


def discover_experiments(configs_dir: Path, glob_expr: str) -> list[ExperimentEntry]:
    files = sorted(p for p in configs_dir.glob(glob_expr) if p.is_file())
    return [ExperimentEntry(key=p.stem, path=p.resolve()) for p in files]


def ensure_state_entries(state: dict[str, Any], experiments: list[ExperimentEntry]) -> None:
    exp_map = state.setdefault("experiments", {})
    now = _utc_now_iso()
    for exp in experiments:
        if exp.key in exp_map:
            exp_map[exp.key]["config_path"] = str(exp.path)
            continue
        exp_map[exp.key] = {
            "config_path": str(exp.path),
            "status": "pending",
            "attempts": 0,
            "created_at": now,
            "updated_at": now,
            "last_exit_code": None,
            "last_started_at": None,
            "last_finished_at": None,
            "last_duration_seconds": None,
            "log_path": None,
        }


def select_queue(
    state: dict[str, Any], experiments: list[ExperimentEntry], retry_failed: bool
) -> list[ExperimentEntry]:
    exp_map: dict[str, Any] = state["experiments"]
    selected: list[ExperimentEntry] = []
    for exp in experiments:
        status = exp_map.get(exp.key, {}).get("status", "pending")
        if status == "completed":
            continue
        if status == "failed" and not retry_failed:
            continue
        selected.append(exp)
    return selected


def _status_counts(state: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {"pending": 0, "running": 0, "completed": 0, "failed": 0, "interrupted": 0}
    for item in state.get("experiments", {}).values():
        status = str(item.get("status", "pending"))
        counts[status] = counts.get(status, 0) + 1
    return counts


def _resolve_python_executable(override: Path | None) -> str:
    if override is not None:
        return str(override)
    repo_root = Path(__file__).resolve().parents[1]
    venv_python = repo_root / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def run_one(
    exp: ExperimentEntry,
    state: dict[str, Any],
    logs_dir: Path,
    python_executable: str,
) -> int:
    exp_state = state["experiments"][exp.key]
    started_at = datetime.now(timezone.utc)
    started_iso = started_at.replace(microsecond=0).isoformat()

    log_path = logs_dir / f"{exp.key}.txt"
    logs_dir.mkdir(parents=True, exist_ok=True)

    exp_state["status"] = "running"
    exp_state["attempts"] = int(exp_state.get("attempts", 0)) + 1
    exp_state["last_started_at"] = started_iso
    exp_state["updated_at"] = started_iso
    exp_state["log_path"] = str(log_path.resolve())

    repo_root = Path(__file__).resolve().parents[1]
    command = [python_executable, "-m", "scripts.run_pipeline", "--config", str(exp.path)]
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    root_str = str(repo_root)
    env["PYTHONPATH"] = (
        f"{root_str}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else root_str
    )
    env["CUDA_VISIBLE_DEVICES"] = "0"

    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"\n[{started_iso}] START {' '.join(command)}\n")
        log_file.flush()

        process = subprocess.Popen(
            command,
            cwd=str(repo_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                print(line, end="")
                log_file.write(line)
            result_code = process.wait()
        except KeyboardInterrupt:
            process.terminate()
            try:
                result_code = process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                result_code = process.wait()
            raise

    finished_at = datetime.now(timezone.utc)
    finished_iso = finished_at.replace(microsecond=0).isoformat()
    duration = (finished_at - started_at).total_seconds()

    exp_state["last_exit_code"] = int(result_code)
    exp_state["last_finished_at"] = finished_iso
    exp_state["last_duration_seconds"] = duration
    exp_state["updated_at"] = finished_iso
    exp_state["status"] = "completed" if result_code == 0 else "failed"

    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(
            f"[{finished_iso}] END exit_code={result_code} duration_seconds={duration:.2f}\n"
        )

    return int(result_code)


def main() -> None:
    args = parse_args()
    configs_dir: Path = args.configs_dir
    state_file: Path = args.state_file
    logs_dir: Path = args.logs_dir
    python_executable = _resolve_python_executable(args.python_executable)

    if not configs_dir.exists():
        raise FileNotFoundError(f"Config directory not found: {configs_dir}")

    experiments = discover_experiments(configs_dir, args.glob)
    if not experiments:
        print(f"[WARN] No configs found in {configs_dir} matching {args.glob}")
        return

    state = load_state(state_file)
    ensure_state_entries(state, experiments)
    state["last_scan_at"] = _utc_now_iso()
    save_state(state_file, state)

    queue = select_queue(state, experiments, retry_failed=args.retry_failed)
    counts = _status_counts(state)
    print(
        "[INFO] Summary:"
        f" total={len(experiments)} pending={counts.get('pending', 0)}"
        f" running={counts.get('running', 0)} completed={counts.get('completed', 0)}"
        f" failed={counts.get('failed', 0)} interrupted={counts.get('interrupted', 0)}"
    )
    print(f"[INFO] Selected to run now: {len(queue)}")
    print(f"[INFO] Python executable: {python_executable}")
    print("[INFO] Forced CUDA_VISIBLE_DEVICES=0")

    if args.dry_run:
        for exp in queue:
            print(f"[DRY-RUN] {exp.key} -> {exp.path}")
        return

    failed_now = 0
    interrupted_now = False

    for idx, exp in enumerate(queue, start=1):
        print(f"[INFO] ({idx}/{len(queue)}) Running {exp.key}")
        try:
            exit_code = run_one(exp, state, logs_dir, python_executable=python_executable)
            save_state(state_file, state)
            if exit_code == 0:
                print(f"[OK] {exp.key}")
            else:
                failed_now += 1
                print(f"[FAIL] {exp.key} (exit_code={exit_code})")
        except KeyboardInterrupt:
            interrupted_now = True
            now = _utc_now_iso()
            exp_state = state["experiments"][exp.key]
            exp_state["status"] = "interrupted"
            exp_state["updated_at"] = now
            exp_state["last_finished_at"] = now
            save_state(state_file, state)
            print(f"[INTERRUPTED] Stopped while running {exp.key}. State saved.")
            break

    final_counts = _status_counts(state)
    print(
        "[INFO] Final summary:"
        f" total={len(experiments)} pending={final_counts.get('pending', 0)}"
        f" running={final_counts.get('running', 0)} completed={final_counts.get('completed', 0)}"
        f" failed={final_counts.get('failed', 0)} interrupted={final_counts.get('interrupted', 0)}"
    )
    print(f"[INFO] State file: {state_file.resolve()}")
    print(f"[INFO] Logs dir: {logs_dir.resolve()}")

    if interrupted_now:
        raise SystemExit(130)
    if failed_now > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
