from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pickle
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config.loader import load_experiment_config

KNOWN_TASKS = [
    "document_retrieval",
    "evidence_retrieval",
    "answer_generation",
]


@dataclass
class ExperimentAudit:
    config_path: Path
    config_stem: str
    experiment_name: str
    dataset_name: str
    chunking_name: str
    routing_name: str
    extrinsic_enabled: bool
    requested_tasks: list[str]
    expected_index_dir: Path
    manifest_path: Path
    metadata_path: Path
    index_path: Path | None
    index_present: bool
    index_details: str
    expected_results_csv: Path
    results_csv_exists: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit extrinsic coverage for a dataset across experiment YAMLs, "
            "find missing indexes/results/tables, and compute missing extrinsic results."
        )
    )
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name, e.g. beir/fiqa")
    parser.add_argument(
        "--configs-dir",
        type=Path,
        required=True,
        help="Experiment configs folder (searched recursively for *.yaml).",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="*.yaml",
        help="Glob expression used recursively under --configs-dir (default: *.yaml).",
    )
    parser.add_argument(
        "--tables-root",
        type=Path,
        default=None,
        help=(
            "Output root for aggregate extrinsic tables. "
            "Default: auto-detect tables_graphs/tables&graphs, fallback tables_graphs."
        ),
    )
    parser.add_argument(
        "--python-executable",
        type=Path,
        default=None,
        help="Python executable used to run missing extrinsic evaluation.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Audit only. Do not execute missing extrinsic evaluations.",
    )
    parser.add_argument(
        "--include-skipped",
        action="store_true",
        help="When rebuilding aggregate tables, include skipped/failed rows.",
    )
    return parser.parse_args()


def _resolve_python_executable(override: Path | None) -> str:
    if override is not None:
        return str(override)
    repo_root = Path(__file__).resolve().parents[1]
    venv_python = repo_root / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def _safe_slug(text: str) -> str:
    value = text.strip().lower()
    for bad, good in [("/", "__"), (" ", "_"), ("-", "_")]:
        value = value.replace(bad, good)
    while "__" in value:
        value = value.replace("__", "_")
    return value.strip("_") or "unknown"


def _resolve_tables_root(arg_tables_root: Path | None) -> Path:
    if arg_tables_root is not None:
        return arg_tables_root
    preferred = Path("tables_graphs")
    legacy = Path("tables&graphs")
    if preferred.exists():
        return preferred
    if legacy.exists():
        return legacy
    return preferred


def _normalize_task_name(task_name: str) -> str:
    return task_name.strip().lower().replace("-", "_")


def _resolve_requested_tasks(config: dict[str, Any]) -> list[str]:
    evaluation_cfg = config.get("evaluation", {})
    requested_tasks = evaluation_cfg.get("extrinsic_tasks_to_run")

    if requested_tasks is None:
        fallback_task = evaluation_cfg.get("extrinsic_evaluator", "document_retrieval")
        requested_tasks = [fallback_task]

    if not isinstance(requested_tasks, list) or not requested_tasks:
        return ["document_retrieval"]

    normalized: list[str] = []
    seen: set[str] = set()
    for task_name in requested_tasks:
        if not isinstance(task_name, str) or not task_name.strip():
            continue
        name = _normalize_task_name(task_name)
        if name not in seen:
            seen.add(name)
            normalized.append(name)

    return normalized or ["document_retrieval"]


def _resolve_output_csv_path(config: dict[str, Any], config_path: Path) -> Path:
    dataset_name = str(config.get("dataset", {}).get("name", "unknown-dataset")).replace("/", "-")
    chunking_name = str(config.get("chunking", {}).get("type", "unknown-chunking"))

    router_cfg = config.get("router", {})
    routing_name = (
        str(router_cfg.get("name", "no-routing")) if bool(router_cfg.get("enabled", False)) else "no-routing"
    )

    experiment_name = (
        config.get("experiment_name")
        or config.get("experiment", {}).get("name")
        or "document-retrieval"
    )

    config_name = config_path.stem.replace("/", "-")

    output_dir = Path(config.get("evaluation", {}).get("results_dir", "results/extrinsic"))
    filename = f"{dataset_name}_{chunking_name}_{routing_name}_{experiment_name}_{config_name}.csv"
    return output_dir / filename


def _resolve_index_dir(config: dict[str, Any], config_path: Path) -> Path:
    retrieval_cfg = config.get("retrieval", {})
    configured_output_dir = retrieval_cfg.get("output_dir")
    if isinstance(configured_output_dir, str) and configured_output_dir.strip():
        return Path(configured_output_dir)

    dataset_name = str(config.get("dataset", {}).get("name", "unknown-dataset"))
    chunking_type = str(config.get("chunking", {}).get("type", "unknown-chunking"))
    experiment_id = config_path.stem
    return Path(f"data/indexes/{dataset_name}/{chunking_type}/{experiment_id}")


def _resolve_index_files(index_dir: Path) -> tuple[Path | None, bool, str]:
    manifest_path = index_dir / "manifest.json"
    metadata_path = index_dir / "metadata.pkl"
    stream_state_path = index_dir / "stream_state.json"

    print(f"[DEBUG] Checking index dir: {index_dir}")

    if not manifest_path.exists() or not metadata_path.exists():
        print(
            "[DEBUG] Missing required files: "
            f"manifest_exists={manifest_path.exists()} metadata_exists={metadata_path.exists()}"
        )
        return None, False, "manifest.json or metadata.pkl missing"

    if not stream_state_path.exists():
        print(f"[DEBUG] Missing stream_state.json: {stream_state_path}")
        return None, False, "stream_state.json missing"

    try:
        stream_state = json.loads(stream_state_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[DEBUG] Failed reading stream_state.json {stream_state_path}: {e}")
        return None, False, f"stream_state unreadable: {e}"

    if not isinstance(stream_state, dict):
        print(f"[DEBUG] stream_state.json is not a JSON object: {stream_state_path}")
        return None, False, "stream_state.json is not a JSON object"

    completed_flag = bool(stream_state.get("completed", False))
    print(f"[DEBUG] stream_state.completed={completed_flag}")
    if not completed_flag:
        return None, False, "stream_state.completed is not true"

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[DEBUG] Failed to read manifest {manifest_path}: {e}")
        return None, False, f"manifest unreadable: {e}"

    if not isinstance(payload, dict):
        print(f"[DEBUG] Manifest is not a JSON object: {manifest_path}")
        return None, False, "manifest is not a JSON object"

    raw_index_path = payload.get("index_path")
    print(f"[DEBUG] manifest.index_path={raw_index_path!r}")
    candidates: list[Path] = []

    if isinstance(raw_index_path, str) and raw_index_path.strip():
        from_manifest = Path(raw_index_path.strip())
        if from_manifest.is_absolute():
            candidates.append(from_manifest)
            # Cross-OS fallback: if absolute path is stale, try same basename in current index_dir.
            candidates.append(index_dir / from_manifest.name)
        else:
            candidates.append(index_dir / from_manifest)

    # Local fallbacks for legacy/cross-machine manifests.
    candidates.extend(
        [
            index_dir / "index.faiss",
            index_dir / "vectors.npy",
        ]
    )

    # Keep first-seen order without duplicates.
    seen: set[Path] = set()
    ordered_candidates: list[Path] = []
    for c in candidates:
        resolved = c.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        ordered_candidates.append(resolved)

    print("[DEBUG] Candidate index paths:")
    for candidate in ordered_candidates:
        print(f"[DEBUG] - {candidate}")

    for candidate in ordered_candidates:
        if candidate.exists():
            if isinstance(raw_index_path, str) and raw_index_path.strip():
                manifest_resolved = Path(raw_index_path.strip())
                if manifest_resolved.is_absolute():
                    manifest_resolved = manifest_resolved.resolve()
                else:
                    manifest_resolved = (index_dir / manifest_resolved).resolve()
                if candidate != manifest_resolved:
                    print(f"[DEBUG] Using fallback index path: {candidate}")
            else:
                print(f"[DEBUG] Using local fallback index path: {candidate}")
            # Validate metadata payload is usable by extrinsic evaluator.
            try:
                with metadata_path.open("rb") as f:
                    metadata_blob = pickle.load(f)
            except Exception as e:
                print(f"[DEBUG] Failed reading metadata.pkl: {e}")
                return candidate, False, f"metadata unreadable: {e}"

            if not isinstance(metadata_blob, dict):
                print("[DEBUG] metadata.pkl is not a dictionary")
                return candidate, False, "metadata.pkl is not a dictionary"

            items = metadata_blob.get("items")
            if isinstance(items, list):
                print(f"[DEBUG] metadata items list found: len={len(items)}")
                return candidate, True, "ok"

            jsonl_raw = metadata_blob.get("items_jsonl_path")
            jsonl_candidates: list[Path] = []
            if isinstance(jsonl_raw, str) and jsonl_raw.strip():
                jsonl_manifest = Path(jsonl_raw.strip())
                if jsonl_manifest.is_absolute():
                    jsonl_candidates.append(jsonl_manifest)
                    jsonl_candidates.append(index_dir / jsonl_manifest.name)
                else:
                    jsonl_candidates.append(index_dir / jsonl_manifest)
                    jsonl_candidates.append(Path(jsonl_raw.strip()))
                    jsonl_candidates.append(Path(jsonl_raw.strip().replace("\\", "/")))
            jsonl_candidates.append(index_dir / "items.jsonl")

            seen_jsonl: set[Path] = set()
            ordered_jsonl: list[Path] = []
            for jc in jsonl_candidates:
                resolved = jc.resolve()
                if resolved in seen_jsonl:
                    continue
                seen_jsonl.add(resolved)
                ordered_jsonl.append(resolved)

            print("[DEBUG] Candidate items.jsonl paths:")
            for jc in ordered_jsonl:
                print(f"[DEBUG] - {jc}")

            for jc in ordered_jsonl:
                if jc.exists():
                    if isinstance(jsonl_raw, str) and jsonl_raw.strip():
                        declared = Path(jsonl_raw.strip())
                        if declared.is_absolute():
                            declared = declared.resolve()
                        else:
                            declared = (index_dir / declared).resolve()
                        if jc != declared:
                            print(f"[DEBUG] Using fallback items.jsonl path: {jc}")
                    else:
                        print(f"[DEBUG] Using local fallback items.jsonl path: {jc}")
                    return candidate, True, "ok"

            print("[DEBUG] metadata has no inlined items and no items.jsonl file was found")
            return candidate, False, "items.jsonl missing for metadata without inlined items"

    if isinstance(raw_index_path, str) and raw_index_path.strip():
        print(f"[DEBUG] No candidate exists. First expected candidate: {ordered_candidates[0]}")
        return ordered_candidates[0], False, f"index file missing: {ordered_candidates[0]}"
    print("[DEBUG] Manifest missing index_path and no local fallback index file found")
    return None, False, "manifest missing index_path and no local fallback index file found"


def _discover_yaml_files(configs_dir: Path, glob_expr: str) -> list[Path]:
    if not configs_dir.exists():
        return []
    return sorted(p for p in configs_dir.rglob(glob_expr) if p.is_file())


def _build_audit_for_config(config_path: Path) -> ExperimentAudit | None:
    try:
        config = load_experiment_config(config_path)
    except Exception as e:
        print(f"[WARN] Skipping unreadable YAML {config_path}: {e}")
        return None

    dataset_name = str(config.get("dataset", {}).get("name", "")).strip()
    if not dataset_name:
        print(f"[WARN] Skipping YAML without dataset.name: {config_path}")
        return None

    requested_tasks = _resolve_requested_tasks(config)
    index_dir = _resolve_index_dir(config, config_path)
    manifest_path = index_dir / "manifest.json"
    metadata_path = index_dir / "metadata.pkl"
    index_path, index_present, index_details = _resolve_index_files(index_dir)

    expected_csv = _resolve_output_csv_path(config, config_path)

    router_cfg = config.get("router", {})
    routing_name = (
        str(router_cfg.get("name", "no-routing")) if bool(router_cfg.get("enabled", False)) else "no-routing"
    )

    experiment_name = (
        str(config.get("experiment_name"))
        if config.get("experiment_name") is not None
        else str(config.get("experiment", {}).get("name", config_path.stem))
    )

    return ExperimentAudit(
        config_path=config_path,
        config_stem=config_path.stem,
        experiment_name=experiment_name,
        dataset_name=dataset_name,
        chunking_name=str(config.get("chunking", {}).get("type", "unknown-chunking")),
        routing_name=routing_name,
        extrinsic_enabled=bool(config.get("evaluation", {}).get("extrinsic", True)),
        requested_tasks=requested_tasks,
        expected_index_dir=index_dir,
        manifest_path=manifest_path,
        metadata_path=metadata_path,
        index_path=index_path,
        index_present=index_present,
        index_details=index_details,
        expected_results_csv=expected_csv,
        results_csv_exists=expected_csv.exists(),
    )


def _print_yaml_summary(audits: list[ExperimentAudit], dataset: str) -> None:
    print("\n=== 1) YAMLs Matching Requested Dataset ===")
    print(f"Dataset: {dataset}")
    print(f"YAML files found: {len(audits)}")
    for item in audits:
        tasks = ",".join(item.requested_tasks)
        print(
            f"- {item.config_path} | chunking={item.chunking_name} | "
            f"experiment={item.experiment_name} | tasks=[{tasks}]"
        )


def _print_index_summary(audits: list[ExperimentAudit]) -> None:
    present = [a for a in audits if a.index_present]
    missing = [a for a in audits if not a.index_present]

    print("\n=== 2) Index Audit (Expected Path and Presence) ===")
    print(f"Indexes available: {len(present)}/{len(audits)}")
    for item in audits:
        idx_label = str(item.index_path) if item.index_path is not None else "n/a"
        print(
            f"- {item.config_stem}: dir={item.expected_index_dir} | "
            f"index={idx_label} | status={'OK' if item.index_present else 'MISSING'} "
            f"({item.index_details})"
        )

    if missing:
        print("\n[INFO] YAML files without a ready index:")
        for item in missing:
            print(f"- {item.config_path}")


def _expected_table_files(dataset: str, tasks: set[str], tables_root: Path) -> list[Path]:
    dataset_slug = _safe_slug(dataset)
    files: list[Path] = []
    for task in sorted(tasks):
        if task not in KNOWN_TASKS:
            continue
        task_dir = tables_root / task / dataset_slug / "tables"
        files.append(task_dir / f"{task}_by_method_all_runs.csv")
        files.append(task_dir / f"{task}_best_per_method.csv")
    return files


def _print_table_summary(audits: list[ExperimentAudit], dataset: str, tables_root: Path) -> list[Path]:
    task_union: set[str] = set()
    for item in audits:
        task_union.update(item.requested_tasks)

    expected_files = _expected_table_files(dataset, task_union, tables_root)
    missing = [p for p in expected_files if not p.exists()]

    print("\n=== 3) Aggregate Extrinsic Tables Audit ===")
    print(f"Tables root: {tables_root}")
    print(f"Expected table files: {len(expected_files)}")
    print(f"Missing table files: {len(missing)}")
    for path in expected_files:
        state = "OK" if path.exists() else "MISSING"
        print(f"- {path} [{state}]")

    indexed_audits = [a for a in audits if a.index_present]
    blocked_no_index = [a for a in audits if not a.index_present]
    missing_csv = [a for a in indexed_audits if not a.results_csv_exists]

    print("\n[INFO] Missing per-YAML extrinsic CSVs (only where index exists):")
    if not missing_csv:
        print("- none")
    else:
        for item in missing_csv:
            print(f"- {item.expected_results_csv} (yaml={item.config_stem})")

    if blocked_no_index:
        print("\n[INFO] YAML files excluded from extrinsic computation (index missing):")
        for item in blocked_no_index:
            print(f"- {item.config_stem}")

    return missing


def _run_missing_extrinsic_evaluations(
    *,
    audits: list[ExperimentAudit],
    python_executable: str,
) -> tuple[list[ExperimentAudit], list[tuple[ExperimentAudit, str]]]:
    completed: list[ExperimentAudit] = []
    failed: list[tuple[ExperimentAudit, str]] = []

    repo_root = Path(__file__).resolve().parents[1]

    for item in audits:
        if item.results_csv_exists:
            continue
        if not item.extrinsic_enabled:
            failed.append((item, "extrinsic disabled in config"))
            continue
        if not item.index_present:
            failed.append((item, f"index missing ({item.index_details})"))
            continue
        if item.index_path is None:
            failed.append((item, "index path unresolved"))
            continue

        item.expected_results_csv.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            python_executable,
            "-m",
            "scripts.run_extrinsic_eval",
            "--config",
            str(item.config_path),
            "--index-path",
            str(item.index_path),
            "--index-metadata-path",
            str(item.metadata_path),
        ]
        if item.manifest_path.exists():
            cmd.extend(["--index-manifest-path", str(item.manifest_path)])

        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH", "")
        root_str = str(repo_root)
        env["PYTHONPATH"] = f"{root_str}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else root_str
        env["CUDA_VISIBLE_DEVICES"] = "0"

        print(f"\n[RUN] {' '.join(cmd)}")
        try:
            subprocess.run(cmd, cwd=str(repo_root), env=env, check=True)
        except subprocess.CalledProcessError as e:
            failed.append((item, f"run_extrinsic_eval failed with exit code {e.returncode}"))
            continue

        if item.expected_results_csv.exists():
            completed.append(item)
        else:
            failed.append((item, "evaluation finished but expected CSV not found"))

    return completed, failed


def _rebuild_dataset_tables(
    *,
    dataset: str,
    tables_root: Path,
    csv_files: list[Path],
    include_skipped: bool,
) -> int:
    if not csv_files:
        return 0

    btg_path = Path(__file__).resolve().parent / "build_tables_graphs.py"
    if not btg_path.exists():
        raise FileNotFoundError(f"build_tables_graphs.py not found at: {btg_path}")

    spec = importlib.util.spec_from_file_location("build_tables_graphs", btg_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec from: {btg_path}")
    btg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(btg)

    data = btg._load_all_rows(csv_files)
    if data.empty:
        return 0

    if not include_skipped:
        skip_values = {"skipped", "failed", "interrupted"}
        status_norm = data["status"].fillna("").astype(str).str.strip().str.lower()
        data = data[(status_norm == "") | (~status_norm.isin(skip_values))].copy()

    if data.empty:
        return 0

    task_count = 0
    summaries: list[dict[str, Any]] = []

    for task_name in btg.TASK_CONFIG.keys():
        part = data[(data["task"] == task_name) & (data["dataset"] == dataset)].copy()
        if part.empty:
            continue
        task_count += 1
        summaries.append(
            btg._write_task_dataset_outputs(
                df_dataset_task=part,
                task_name=task_name,
                dataset_name=dataset,
                output_root=tables_root,
                save_pdf=False,
            )
        )

    global_dir = tables_root / "global"
    global_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd

    pd.DataFrame(summaries).to_csv(global_dir / "output_summary.csv", index=False)
    pd.DataFrame([{"source_csv": str(p)} for p in csv_files]).to_csv(
        global_dir / "run_metadata.csv", index=False
    )

    mapping = {
        "tasks": btg.TASK_CONFIG,
        "method_plot_order": btg.METHOD_PLOT_ORDER,
        "status_filtering": {
            "include_skipped": bool(include_skipped),
            "excluded_if_false": ["skipped", "failed", "interrupted"],
        },
    }
    (global_dir / "column_mapping_used.json").write_text(
        json.dumps(mapping, indent=2),
        encoding="utf-8",
    )

    return task_count


def main() -> None:
    args = parse_args()

    dataset = args.dataset.strip()
    if not dataset:
        raise ValueError("--dataset cannot be empty")

    if not args.configs_dir.exists():
        raise FileNotFoundError(f"Configs directory not found: {args.configs_dir}")

    tables_root = _resolve_tables_root(args.tables_root)
    yaml_files = _discover_yaml_files(args.configs_dir, args.glob)
    if not yaml_files:
        print(f"[WARN] No YAML files found in {args.configs_dir} with glob {args.glob}")
        return

    audits: list[ExperimentAudit] = []
    for config_path in yaml_files:
        item = _build_audit_for_config(config_path)
        if item is None:
            continue
        if item.dataset_name != dataset:
            continue
        audits.append(item)

    audits = sorted(audits, key=lambda x: x.config_path)

    if not audits:
        print(f"[WARN] No experiment YAML found for dataset '{dataset}' in {args.configs_dir}")
        return

    _print_yaml_summary(audits, dataset)
    _print_index_summary(audits)
    _print_table_summary(audits, dataset, tables_root)

    indexed_audits = [a for a in audits if a.index_present]
    missing_results = [a for a in indexed_audits if not a.results_csv_exists]
    computed_this_run = 0

    print("\n=== 4) Backfill Missing Extrinsic Metrics ===")
    if not missing_results:
        print("No missing extrinsic CSVs: nothing to compute.")
    elif args.dry_run:
        print("Dry-run enabled: skipping run_extrinsic_eval execution.")
    else:
        python_executable = _resolve_python_executable(args.python_executable)
        completed, failed = _run_missing_extrinsic_evaluations(
            audits=indexed_audits,
            python_executable=python_executable,
        )
        computed_this_run = len(completed)

        print(f"\n[SUMMARY] Completed extrinsic backfills: {len(completed)}")
        print(f"[SUMMARY] Failed/skipped extrinsic backfills: {len(failed)}")
        for item, reason in failed:
            print(f"- {item.config_stem}: {reason}")

    # Refresh in-memory existence after optional backfill.
    for idx, item in enumerate(audits):
        audits[idx].results_csv_exists = item.expected_results_csv.exists()

    dataset_csvs = sorted(
        {
            item.expected_results_csv.resolve()
            for item in audits
            if item.index_present and item.expected_results_csv.exists()
        }
    )

    print("\n=== 5) Rebuild Dataset Aggregate Tables ===")
    if not dataset_csvs:
        print("No extrinsic CSVs available: tables were not updated.")
    else:
        generated_tasks = _rebuild_dataset_tables(
            dataset=dataset,
            tables_root=tables_root,
            csv_files=dataset_csvs,
            include_skipped=bool(args.include_skipped),
        )
        print(f"Tables updated in: {tables_root.resolve()}")
        print(f"Tasks updated: {generated_tasks}")

    print("\n=== Final Summary ===")
    print(f"number total experiments: {len(audits)}")
    print(f"number indexes found: {len(indexed_audits)}")
    print(f"number of extrinsic evaluation missing w.r.t indexes found: {len(missing_results)}")
    print(f"number of extrinsic computed in this run: {computed_this_run}")


if __name__ == "__main__":
    main()
