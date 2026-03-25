from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

# Keep matplotlib cache writable also in constrained VM/home setups.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TASK_CONFIG: dict[str, dict[str, Any]] = {
    "document_retrieval": {
        "metrics": ["f1", "ndcg"],
        "ks": [1, 3, 5, 10],
        "task_dir": "document_retrieval",
    },
    "evidence_retrieval": {
        "metrics": ["f1"],
        "ks": [1, 3, 5, 10],
        "task_dir": "evidence_retrieval",
    },
    "answer_generation": {
        "metrics": ["qa_similarity", "bertscore_f1"],
        "ks": [1, 3, 5, 10],
        "task_dir": "answer_generation",
    },
}

METHOD_PLOT_ORDER = [
    "fixed",
    "semantic_breakpoint",
    "clustering_dbscan",
    "clustering_single_linkage",
    "clustering_all",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build summary tables and boxplots for extrinsic results "
            "(document retrieval, evidence retrieval, answer generation)."
        )
    )
    parser.add_argument(
        "--input-glob",
        action="append",
        default=["results/extrinsic/*.csv"],
        help=(
            "Glob expression for result CSV files. Can be provided multiple times. "
            "Default: results/extrinsic/*.csv"
        ),
    )
    parser.add_argument(
        "--input-file",
        action="append",
        default=[],
        help="Single CSV file path to include. Can be provided multiple times.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tables&graphs"),
        help="Output directory for generated tables and plots.",
    )
    parser.add_argument(
        "--include-skipped",
        action="store_true",
        help="Include rows with skipped/failed status (normally excluded).",
    )
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Do not export PDF copies of plots.",
    )
    return parser.parse_args()


def _collect_input_csvs(globs: list[str], files: list[str]) -> list[Path]:
    collected: set[Path] = set()
    for pattern in globs:
        for path in Path(".").glob(pattern):
            if path.is_file() and path.suffix.lower() == ".csv":
                collected.add(path.resolve())

    for file_path in files:
        p = Path(file_path).expanduser().resolve()
        if not p.exists() or not p.is_file():
            raise FileNotFoundError(f"Input file not found: {p}")
        if p.suffix.lower() != ".csv":
            raise ValueError(f"Input file is not a CSV: {p}")
        collected.add(p)

    return sorted(collected)


def _safe_slug(text: str) -> str:
    value = text.strip().lower()
    for bad, good in [("/", "__"), (" ", "_"), ("-", "_")]:
        value = value.replace(bad, good)
    while "__" in value:
        value = value.replace("__", "_")
    return value.strip("_") or "unknown"


def _metric_label(metric: str, k: int | None = None) -> str:
    name_map = {
        "f1": "F1",
        "ndcg": "NDCG",
        "qa_similarity": "CosineSimilarity",
        "bertscore_f1": "BERTScoreF1",
    }
    base = name_map.get(metric, metric)
    return f"{base}@{k}" if k is not None else base


def _infer_method(row: pd.Series) -> tuple[str, str]:
    source_stem = str(row.get("source_stem", ""))
    chunking = str(row.get("chunking", "")).strip().lower()
    hint = f"{source_stem} {chunking}".upper()

    if "DBSCAN" in hint:
        return "clustering", "clustering_dbscan"
    if "SLINK" in hint or "SINGLE_LINKAGE" in hint:
        return "clustering", "clustering_single_linkage"
    if "FIXED" in hint or chunking == "fixed":
        return "fixed", "fixed"
    if "BREAK" in hint or chunking == "semantic_breakpoint":
        return "semantic_breakpoint", "semantic_breakpoint"
    if "CLUSTER" in hint or chunking in {"semantic_clustering", "clustering"}:
        return "clustering", "clustering_unknown"

    if chunking:
        return chunking, chunking
    return "unknown", "unknown"


def _load_all_rows(csv_paths: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in csv_paths:
        try:
            df = pd.read_csv(path)
        except Exception as e:
            raise RuntimeError(f"Failed reading CSV: {path} ({e})") from e

        if df.empty:
            continue

        df["source_csv"] = str(path)
        df["source_stem"] = path.stem
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    data = pd.concat(frames, axis=0, ignore_index=True)

    for c in ["k", "f1", "ndcg", "qa_similarity", "bertscore_f1"]:
        if c in data.columns:
            data[c] = pd.to_numeric(data[c], errors="coerce")

    if "status" not in data.columns:
        data["status"] = ""

    data["dataset"] = data.get("dataset", "unknown").fillna("unknown").astype(str)
    data["task"] = data.get("task", "unknown").fillna("unknown").astype(str)

    methods = data.apply(_infer_method, axis=1, result_type="expand")
    data["method_macro"] = methods[0]
    data["method_variant"] = methods[1]

    return data


def _prepare_run_wide_table(df_task: pd.DataFrame, metrics: list[str], ks: list[int]) -> pd.DataFrame:
    id_cols = [
        "dataset",
        "method_macro",
        "method_variant",
        "chunking",
        "routing",
        "experiment",
        "retriever",
        "embedder",
        "source_stem",
        "source_csv",
    ]
    for col in id_cols:
        if col not in df_task.columns:
            df_task[col] = ""

    wide = df_task[id_cols].drop_duplicates().copy()
    for metric in metrics:
        if metric not in df_task.columns:
            continue
        for k in ks:
            mask = (df_task["k"] == int(k))
            subset = df_task.loc[mask, id_cols + [metric]].copy()
            if subset.empty:
                continue
            col_name = _metric_label(metric, k)
            agg = subset.groupby(id_cols, as_index=False)[metric].mean().rename(columns={metric: col_name})
            wide = wide.merge(agg, on=id_cols, how="left")

    sort_cols = ["method_variant", "source_stem"]
    return wide.sort_values(sort_cols).reset_index(drop=True)


def _build_best_per_method_table(run_wide: pd.DataFrame, metric_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if run_wide.empty:
        return pd.DataFrame()

    for method in sorted(run_wide["method_variant"].dropna().astype(str).unique()):
        part = run_wide[run_wide["method_variant"] == method]
        row: dict[str, Any] = {"method_variant": method}
        for mc in metric_cols:
            if mc not in part.columns:
                continue
            s = pd.to_numeric(part[mc], errors="coerce")
            if s.dropna().empty:
                row[f"best_{mc}"] = np.nan
                row[f"best_{mc}_run"] = ""
                continue
            best_idx = s.idxmax()
            row[f"best_{mc}"] = float(s.loc[best_idx])
            row[f"best_{mc}_run"] = str(part.loc[best_idx, "source_stem"])
        rows.append(row)

    return pd.DataFrame(rows).sort_values("method_variant").reset_index(drop=True)


def _save_boxplot(
    run_wide: pd.DataFrame,
    dataset_name: str,
    metric_cols: list[str],
    output_png: Path,
    output_pdf: Path | None,
) -> None:
    if run_wide.empty or not metric_cols:
        return

    n_metrics = len(metric_cols)
    n_cols = 4 if n_metrics >= 4 else n_metrics
    n_rows = int(np.ceil(n_metrics / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4.2 * n_rows), squeeze=False)
    fig.suptitle(f"{dataset_name} - Method Comparison", fontsize=14)

    for idx, metric_col in enumerate(metric_cols):
        r = idx // n_cols
        c = idx % n_cols
        ax = axes[r][c]

        grouped_values: list[np.ndarray] = []
        for method in METHOD_PLOT_ORDER:
            if method == "clustering_all":
                values = pd.to_numeric(
                    run_wide.loc[run_wide["method_macro"] == "clustering", metric_col],
                    errors="coerce",
                ).dropna()
            else:
                values = pd.to_numeric(
                    run_wide.loc[run_wide["method_variant"] == method, metric_col],
                    errors="coerce",
                ).dropna()

            arr = values.to_numpy(dtype=float)
            grouped_values.append(arr if arr.size > 0 else np.array([np.nan]))

        bp = ax.boxplot(grouped_values, labels=METHOD_PLOT_ORDER, patch_artist=True, showfliers=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("#8ecae6")
            patch.set_alpha(0.7)

        # Overlay points with tiny jitter for transparency on spread.
        for pos, values in enumerate(grouped_values, start=1):
            clean = values[~np.isnan(values)]
            if clean.size == 0:
                continue
            jitter = np.random.uniform(-0.12, 0.12, size=clean.size)
            ax.scatter(np.full(clean.shape, pos) + jitter, clean, s=15, alpha=0.5, color="#023047")

        ax.set_title(metric_col)
        ax.set_xticklabels(METHOD_PLOT_ORDER, rotation=18, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.25)

    for idx in range(n_metrics, n_rows * n_cols):
        r = idx // n_cols
        c = idx % n_cols
        axes[r][c].axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=220)
    if output_pdf is not None:
        fig.savefig(output_pdf)
    plt.close(fig)


def _write_task_dataset_outputs(
    *,
    df_dataset_task: pd.DataFrame,
    task_name: str,
    dataset_name: str,
    output_root: Path,
    save_pdf: bool,
) -> dict[str, Any]:
    cfg = TASK_CONFIG[task_name]
    task_dir = output_root / cfg["task_dir"] / _safe_slug(dataset_name)
    tables_dir = task_dir / "tables"
    plots_dir = task_dir / "plots"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    run_wide = _prepare_run_wide_table(
        df_task=df_dataset_task.copy(),
        metrics=cfg["metrics"],
        ks=cfg["ks"],
    )

    metric_cols = [
        _metric_label(metric, k)
        for metric in cfg["metrics"]
        for k in cfg["ks"]
        if _metric_label(metric, k) in run_wide.columns
    ]

    by_method_path = tables_dir / f"{task_name}_by_method_all_runs.csv"
    run_wide.to_csv(by_method_path, index=False)

    best_table = _build_best_per_method_table(run_wide, metric_cols)
    best_path = tables_dir / f"{task_name}_best_per_method.csv"
    best_table.to_csv(best_path, index=False)

    boxplot_png = plots_dir / f"{task_name}_boxplot_methods.png"
    boxplot_pdf = plots_dir / f"{task_name}_boxplot_methods.pdf" if save_pdf else None
    _save_boxplot(
        run_wide=run_wide,
        dataset_name=dataset_name,
        metric_cols=metric_cols,
        output_png=boxplot_png,
        output_pdf=boxplot_pdf,
    )

    return {
        "dataset": dataset_name,
        "task": task_name,
        "rows": int(len(df_dataset_task)),
        "runs": int(len(run_wide)),
        "metrics": metric_cols,
        "by_method_csv": str(by_method_path),
        "best_csv": str(best_path),
        "plot_png": str(boxplot_png),
        "plot_pdf": str(boxplot_pdf) if boxplot_pdf is not None else None,
    }


def main() -> None:
    args = parse_args()
    input_csvs = _collect_input_csvs(args.input_glob, args.input_file)
    if not input_csvs:
        raise FileNotFoundError(
            "No input CSV files found. Pass --input-file and/or --input-glob."
        )

    data = _load_all_rows(input_csvs)
    if data.empty:
        raise RuntimeError("No rows found in provided CSV files.")

    if not args.include_skipped:
        skip_values = {"skipped", "failed", "interrupted"}
        status_norm = data["status"].fillna("").astype(str).str.strip().str.lower()
        data = data[(status_norm == "") | (~status_norm.isin(skip_values))].copy()

    output_root: Path = args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    for task_name in TASK_CONFIG.keys():
        task_df = data[data["task"] == task_name].copy()
        if task_df.empty:
            continue
        for dataset_name, part in task_df.groupby("dataset"):
            summaries.append(
                _write_task_dataset_outputs(
                    df_dataset_task=part,
                    task_name=task_name,
                    dataset_name=str(dataset_name),
                    output_root=output_root,
                    save_pdf=not args.no_pdf,
                )
            )

    global_dir = output_root / "global"
    global_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [{"source_csv": str(p)} for p in input_csvs]
    ).to_csv(global_dir / "run_metadata.csv", index=False)

    mapping = {
        "tasks": TASK_CONFIG,
        "method_plot_order": METHOD_PLOT_ORDER,
        "method_inference": {
            "dbscan": "clustering_dbscan",
            "slink/single_linkage": "clustering_single_linkage",
            "fixed": "fixed",
            "break/semantic_breakpoint": "semantic_breakpoint",
            "cluster fallback": "clustering_unknown",
        },
        "status_filtering": {
            "include_skipped": bool(args.include_skipped),
            "excluded_if_false": ["skipped", "failed", "interrupted"],
        },
    }
    (global_dir / "column_mapping_used.json").write_text(
        json.dumps(mapping, indent=2),
        encoding="utf-8",
    )

    pd.DataFrame(summaries).to_csv(global_dir / "output_summary.csv", index=False)

    print(f"[INFO] Input CSV files: {len(input_csvs)}")
    print(f"[INFO] Output directory: {output_root.resolve()}")
    print(f"[INFO] Generated dataset/task outputs: {len(summaries)}")


if __name__ == "__main__":
    main()
