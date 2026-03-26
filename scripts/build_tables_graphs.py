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
from matplotlib.backends.backend_pdf import PdfPages


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

INTRINSIC_METRIC_PLOT_ORDER = [
    "boundary_clarity",
    "chunk_stickiness_complete",
    "chunk_stickiness_incomplete",
]

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
            "Build summary tables and boxplots for extrinsic and intrinsic results "
            "(document retrieval, evidence retrieval, answer generation, chunk quality)."
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
        "--intrinsic-input-glob",
        action="append",
        default=["results/intrinsic/*.json"],
        help=(
            "Glob expression for intrinsic JSON files. Can be provided multiple times. "
            "Default: results/intrinsic/*.json"
        ),
    )
    parser.add_argument(
        "--intrinsic-input-file",
        action="append",
        default=[],
        help="Single intrinsic JSON file path to include. Can be provided multiple times.",
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


def _collect_input_jsons(globs: list[str], files: list[str]) -> list[Path]:
    collected: set[Path] = set()
    for pattern in globs:
        for path in Path(".").glob(pattern):
            if path.is_file() and path.suffix.lower() == ".json":
                if path.stem.endswith("_global"):
                    continue
                collected.add(path.resolve())

    for file_path in files:
        p = Path(file_path).expanduser().resolve()
        if not p.exists() or not p.is_file():
            raise FileNotFoundError(f"Intrinsic input file not found: {p}")
        if p.suffix.lower() != ".json":
            raise ValueError(f"Intrinsic input file is not a JSON: {p}")
        if p.stem.endswith("_global"):
            continue
        collected.add(p)

    return sorted(collected)


def _safe_slug(text: str) -> str:
    value = text.strip().lower()
    for bad, good in [("/", "__"), (" ", "_"), ("-", "_")]:
        value = value.replace(bad, good)
    while "__" in value:
        value = value.replace("__", "_")
    return value.strip("_") or "unknown"


def _build_dataset_task_dirs(output_root: Path, dataset_name: str, task_segment: str) -> tuple[Path, Path]:
    dataset_dir = output_root / _safe_slug(dataset_name)
    task_dir = dataset_dir / _safe_slug(task_segment)
    tables_dir = task_dir / "tables"
    plots_dir = task_dir / "plots"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    return tables_dir, plots_dir


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


def _summarize_per_document_metrics(per_doc: Any) -> dict[str, float]:
    if not isinstance(per_doc, list):
        return {}

    values_by_key: dict[str, list[float]] = {}
    for row in per_doc:
        if not isinstance(row, dict):
            continue
        for key, value in row.items():
            if key == "doc_id":
                continue
            if isinstance(value, (int, float)):
                values_by_key.setdefault(key, []).append(float(value))

    summary: dict[str, float] = {}
    for key, values in values_by_key.items():
        if not values:
            continue
        summary[f"per_doc_mean_{key}"] = float(sum(values) / len(values))

    return summary


def _load_intrinsic_rows(json_paths: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in json_paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        if not isinstance(payload, dict):
            continue

        metrics = payload.get("metrics", {})
        metadata = payload.get("metadata", {})
        row: dict[str, Any] = {
            "source_json": str(path),
            "source_stem": path.stem,
            "dataset": _infer_intrinsic_dataset_from_stem(path.stem),
        }

        if isinstance(metrics, dict):
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    row[key] = float(value)

        if isinstance(metadata, dict):
            for key in ["num_documents", "num_chunks"]:
                value = metadata.get(key)
                if isinstance(value, (int, float)):
                    row[key] = float(value)

        row.update(_summarize_per_document_metrics(payload.get("per_document_metrics")))

        methods = _infer_method(pd.Series(row))
        row["method_macro"] = methods[0]
        row["method_variant"] = methods[1]
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


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


def _save_dataframe_pdf_table(
    df: pd.DataFrame,
    output_pdf: Path,
    title: str,
    pinned_columns: list[str] | None = None,
) -> None:
    if df.empty:
        return

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    pinned_columns = pinned_columns or []

    existing_pinned = [c for c in pinned_columns if c in df.columns]
    other_cols = [c for c in df.columns if c not in existing_pinned]

    max_other_cols_per_page = 6
    col_chunks: list[list[str]] = []
    if not other_cols:
        col_chunks = [existing_pinned]
    else:
        for i in range(0, len(other_cols), max_other_cols_per_page):
            col_chunks.append(existing_pinned + other_cols[i : i + max_other_cols_per_page])

    rows_per_page = 24

    with PdfPages(output_pdf) as pdf:
        total_pages = int(np.ceil(len(df) / rows_per_page)) * len(col_chunks)
        page_no = 0
        for chunk_idx, cols in enumerate(col_chunks, start=1):
            chunk_df = df.loc[:, cols].copy()
            n_row_pages = int(np.ceil(len(chunk_df) / rows_per_page))
            for row_page in range(n_row_pages):
                page_no += 1
                start = row_page * rows_per_page
                end = start + rows_per_page
                page_df = chunk_df.iloc[start:end].copy()

                n_cols = max(1, len(page_df.columns))
                fig_w = min(24, max(12, 2.1 * n_cols))
                fig_h = min(15, max(6, 1.4 + 0.28 * (len(page_df) + 2)))
                fig, ax = plt.subplots(figsize=(fig_w, fig_h))
                ax.axis("off")
                ax.set_title(
                    (
                        f"{title} "
                        f"(page {page_no}/{total_pages}, col-block {chunk_idx}/{len(col_chunks)})"
                    ),
                    fontsize=11,
                    pad=10,
                )

                table = ax.table(
                    cellText=page_df.astype(str).values,
                    colLabels=[str(c) for c in page_df.columns],
                    loc="center",
                    cellLoc="left",
                )
                table.auto_set_font_size(False)
                if n_cols <= 5:
                    font_size = 9
                elif n_cols <= 8:
                    font_size = 8
                else:
                    font_size = 7
                table.set_fontsize(font_size)
                table.scale(1.0, 1.2)

                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)


def _infer_intrinsic_dataset_from_stem(source_stem: str) -> str:
    tokens = [t for t in source_stem.split("_") if t]
    if not tokens:
        return "unknown"

    markers = {"fixed", "break", "semantic", "dbscan", "slink", "clustering"}
    marker_idx = None
    for idx, tok in enumerate(tokens):
        if tok.lower() in markers:
            marker_idx = idx
            break

    if marker_idx is None:
        return tokens[0]
    if marker_idx == 0:
        return "unknown"
    return "_".join(tokens[:marker_idx])


def _build_extrinsic_long_table(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if data.empty:
        return pd.DataFrame()

    for task_name, cfg in TASK_CONFIG.items():
        task_df = data[data["task"] == task_name].copy()
        if task_df.empty:
            continue
        for metric in cfg["metrics"]:
            if metric not in task_df.columns:
                continue
            for k in cfg["ks"]:
                part = task_df[task_df["k"] == int(k)].copy()
                if part.empty:
                    continue
                values = pd.to_numeric(part[metric], errors="coerce")
                part = part.loc[values.notna()].copy()
                if part.empty:
                    continue
                part["metric"] = metric
                part["metric_label"] = _metric_label(metric, k)
                part["k"] = int(k)
                part["value"] = pd.to_numeric(part[metric], errors="coerce")
                rows.extend(
                    {
                        "origin": "extrinsic",
                        "task": task_name,
                        "dataset": str(r.get("dataset", "unknown")),
                        "method_macro": str(r.get("method_macro", "")),
                        "method_variant": str(r.get("method_variant", "")),
                        "metric": str(r["metric"]),
                        "metric_label": str(r["metric_label"]),
                        "k": int(r["k"]),
                        "value": float(r["value"]),
                        "source_stem": str(r.get("source_stem", "")),
                        "source_path": str(r.get("source_csv", "")),
                    }
                    for _, r in part.iterrows()
                )

    return pd.DataFrame(rows)


def _build_intrinsic_long_table(intrinsic_data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if intrinsic_data.empty:
        return pd.DataFrame()

    exclude_cols = {
        "source_json",
        "source_stem",
        "method_macro",
        "method_variant",
    }

    for _, r in intrinsic_data.iterrows():
        dataset_name = str(r.get("dataset", "")) or _infer_intrinsic_dataset_from_stem(str(r.get("source_stem", "")))
        for col in intrinsic_data.columns:
            if col in exclude_cols:
                continue
            value = pd.to_numeric(pd.Series([r[col]]), errors="coerce").iloc[0]
            if pd.isna(value):
                continue
            if col.startswith("per_doc_mean_"):
                metric_group = "per_doc_mean"
            elif col in {"num_documents", "num_chunks"}:
                metric_group = "metadata"
            else:
                metric_group = "aggregate"
            rows.append(
                {
                    "origin": "intrinsic",
                    "task": "intrinsic",
                    "dataset": dataset_name,
                    "method_macro": str(r.get("method_macro", "")),
                    "method_variant": str(r.get("method_variant", "")),
                    "metric_group": metric_group,
                    "metric": str(col),
                    "metric_label": str(col),
                    "k": np.nan,
                    "value": float(value),
                    "source_stem": str(r.get("source_stem", "")),
                    "source_path": str(r.get("source_json", "")),
                }
            )

    return pd.DataFrame(rows)


def _save_intrinsic_outputs(
    intrinsic_data: pd.DataFrame,
    output_root: Path,
    save_pdf: bool,
) -> list[dict[str, Any]]:
    if intrinsic_data.empty:
        return []

    summaries: list[dict[str, Any]] = []
    for dataset_name, part in intrinsic_data.groupby("dataset"):
        run_table = part.sort_values(["method_variant", "source_stem"]).reset_index(drop=True)
        tables_dir, plots_dir = _build_dataset_task_dirs(
            output_root=output_root,
            dataset_name=str(dataset_name),
            task_segment="intrinsic",
        )

        by_run_path = tables_dir / "intrinsic_by_method_all_runs.csv"
        run_table.to_csv(by_run_path, index=False)
        by_run_pdf = tables_dir / "intrinsic_by_method_all_runs.pdf"
        _save_dataframe_pdf_table(
            run_table,
            by_run_pdf,
            f"{dataset_name} - intrinsic by run",
            pinned_columns=["dataset", "method_variant", "source_stem"],
        )

        available_metric_cols = [c for c in INTRINSIC_METRIC_PLOT_ORDER if c in run_table.columns]
        best_table = _build_best_per_method_table(run_table, available_metric_cols)
        best_path = tables_dir / "intrinsic_best_per_method.csv"
        best_table.to_csv(best_path, index=False)
        best_pdf = tables_dir / "intrinsic_best_per_method.pdf"
        _save_dataframe_pdf_table(
            best_table,
            best_pdf,
            f"{dataset_name} - intrinsic best per method",
            pinned_columns=["method_variant"],
        )

        boxplot_png = plots_dir / "intrinsic_boxplot_methods.png"
        boxplot_pdf = plots_dir / "intrinsic_boxplot_methods.pdf" if save_pdf else None
        _save_boxplot(
            run_wide=run_table,
            dataset_name=f"{dataset_name} - intrinsic",
            metric_cols=available_metric_cols,
            output_png=boxplot_png,
            output_pdf=boxplot_pdf,
        )

        summaries.append(
            {
                "dataset": str(dataset_name),
                "task": "intrinsic",
                "rows": int(len(run_table)),
                "runs": int(len(run_table)),
                "metrics": available_metric_cols,
                "by_method_csv": str(by_run_path),
                "by_method_pdf": str(by_run_pdf),
                "best_csv": str(best_path),
                "best_pdf": str(best_pdf),
                "plot_png": str(boxplot_png),
                "plot_pdf": str(boxplot_pdf) if boxplot_pdf is not None else None,
            }
        )

    return summaries


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
    tables_dir, plots_dir = _build_dataset_task_dirs(
        output_root=output_root,
        dataset_name=dataset_name,
        task_segment=cfg["task_dir"],
    )

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
    by_method_pdf = tables_dir / f"{task_name}_by_method_all_runs.pdf"
    _save_dataframe_pdf_table(
        run_wide,
        by_method_pdf,
        f"{dataset_name} - {task_name} by run",
        pinned_columns=["dataset", "method_variant", "source_stem"],
    )

    best_table = _build_best_per_method_table(run_wide, metric_cols)
    best_path = tables_dir / f"{task_name}_best_per_method.csv"
    best_table.to_csv(best_path, index=False)
    best_pdf = tables_dir / f"{task_name}_best_per_method.pdf"
    _save_dataframe_pdf_table(
        best_table,
        best_pdf,
        f"{dataset_name} - {task_name} best per method",
        pinned_columns=["method_variant"],
    )

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
        "by_method_pdf": str(by_method_pdf),
        "best_csv": str(best_path),
        "best_pdf": str(best_pdf),
        "plot_png": str(boxplot_png),
        "plot_pdf": str(boxplot_pdf) if boxplot_pdf is not None else None,
    }


def main() -> None:
    args = parse_args()
    input_csvs = _collect_input_csvs(args.input_glob, args.input_file)
    intrinsic_jsons = _collect_input_jsons(args.intrinsic_input_glob, args.intrinsic_input_file)
    if not input_csvs and not intrinsic_jsons:
        raise FileNotFoundError("No input files found for extrinsic CSV and intrinsic JSON.")

    data = _load_all_rows(input_csvs) if input_csvs else pd.DataFrame()
    intrinsic_data = _load_intrinsic_rows(intrinsic_jsons) if intrinsic_jsons else pd.DataFrame()
    if data.empty and intrinsic_data.empty:
        raise RuntimeError("No valid rows found in provided files.")

    if not data.empty and not args.include_skipped:
        skip_values = {"skipped", "failed", "interrupted"}
        status_norm = data["status"].fillna("").astype(str).str.strip().str.lower()
        data = data[(status_norm == "") | (~status_norm.isin(skip_values))].copy()

    output_root: Path = args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    if not data.empty:
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

    intrinsic_summaries = _save_intrinsic_outputs(
        intrinsic_data=intrinsic_data,
        output_root=output_root,
        save_pdf=not args.no_pdf,
    )
    summaries.extend(intrinsic_summaries)

    global_dir = output_root / "global"
    global_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        (
            [{"source_csv": str(p)} for p in input_csvs]
            + [{"source_json": str(p)} for p in intrinsic_jsons]
        )
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
        "intrinsic_metrics_plot_order": INTRINSIC_METRIC_PLOT_ORDER,
        "output_hierarchy": "<output_dir>/<dataset>/<task>/[tables|plots]",
    }
    (global_dir / "column_mapping_used.json").write_text(
        json.dumps(mapping, indent=2),
        encoding="utf-8",
    )

    pd.DataFrame(summaries).to_csv(global_dir / "output_summary.csv", index=False)

    extrinsic_long = _build_extrinsic_long_table(data)
    intrinsic_long = _build_intrinsic_long_table(intrinsic_data)
    metrics_long = pd.concat([extrinsic_long, intrinsic_long], axis=0, ignore_index=True)
    if not metrics_long.empty:
        metrics_long = metrics_long.sort_values(
            ["dataset", "task", "metric", "k", "method_variant", "source_stem"],
            na_position="last",
        ).reset_index(drop=True)
    metrics_long.to_csv(global_dir / "metrics_long.csv", index=False)

    print(f"[INFO] Input CSV files: {len(input_csvs)}")
    print(f"[INFO] Input intrinsic JSON files: {len(intrinsic_jsons)}")
    print(f"[INFO] Output directory: {output_root.resolve()}")
    print(f"[INFO] Generated dataset/task outputs: {len(summaries)}")
    print(f"[INFO] Rows in global metrics_long.csv: {len(metrics_long)}")


if __name__ == "__main__":
    main()
