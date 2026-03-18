from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from src.datasets.base import BaseDatasetProcessor


class _ScriptedNormalizedDatasetProcessor(BaseDatasetProcessor):
    """
    Processor for datasets that are produced directly in normalized format
    through project scripts (no BEIR raw folder expected).
    """

    script_name: str = ""

    def should_save_normalized(self, config: dict) -> bool:
        # The converter script already writes normalized files.
        return False

    def should_delete_raw_after_normalized(self, config: dict) -> bool:
        # There is no separate raw folder in this workflow.
        return False

    def ensure_available(self, config: dict) -> Path:
        split = self.get_split(config)
        normalized_dir = self.resolve_normalized_dir(config)
        if self.normalized_exists(normalized_dir, split):
            return normalized_dir

        if not self.should_download(config):
            raise FileNotFoundError(
                "Normalized dataset missing and download_if_missing=false. "
                f"Expected normalized cache at: {normalized_dir}"
            )

        self._run_converter_script(config=config, normalized_dir=normalized_dir, split=split)

        if not self.normalized_exists(normalized_dir, split):
            raise FileNotFoundError(
                "Converter script completed but normalized dataset is incomplete. "
                f"Expected split '{split}' at: {normalized_dir}"
            )

        return normalized_dir

    def load_raw(self, raw_path: Path, config: dict) -> Any:
        # `raw_path` is the normalized dir for this scripted processor.
        split = self.get_split(config)
        return self.load_normalized(normalized_dir=raw_path, split=split, config=config)

    def _run_converter_script(self, config: dict, normalized_dir: Path, split: str) -> None:
        project_root = Path(__file__).resolve().parents[2]
        script_path = project_root / "scripts" / self.script_name
        if not script_path.exists():
            raise FileNotFoundError(f"Converter script not found: {script_path}")

        cmd = [
            sys.executable,
            str(script_path),
            "--source-split",
            str(config.get("source_split", split)),
            "--output-split",
            split,
            "--target-dir",
            str(normalized_dir),
        ]
        cmd.extend(self._extra_cli_args(config))

        print(
            "[DATASET] Building normalized cache via converter script: "
            f"{self.script_name} (split={split})"
        )
        subprocess.run(cmd, cwd=project_root, check=True)

    def _extra_cli_args(self, config: dict) -> list[str]:
        return []


class QasperScriptedProcessor(_ScriptedNormalizedDatasetProcessor):
    script_name = "download_qasper_hf_to_normalized.py"

    def _extra_cli_args(self, config: dict) -> list[str]:
        out: list[str] = []
        hf_dataset = config.get("hf_dataset")
        if isinstance(hf_dataset, str) and hf_dataset.strip():
            out.extend(["--hf-dataset", hf_dataset.strip()])

        max_documents = config.get("max_documents")
        if isinstance(max_documents, int) and max_documents > 0:
            out.extend(["--max-documents", str(max_documents)])

        max_chars = config.get("max_chars_per_doc")
        if isinstance(max_chars, int) and max_chars > 0:
            out.extend(["--max-chars-per-doc", str(max_chars)])

        return out


class TechQAScriptedProcessor(_ScriptedNormalizedDatasetProcessor):
    script_name = "download_techqa_hf_to_normalized.py"

    def _extra_cli_args(self, config: dict) -> list[str]:
        out: list[str] = []
        hf_dataset = config.get("hf_dataset")
        if isinstance(hf_dataset, str) and hf_dataset.strip():
            out.extend(["--hf-dataset", hf_dataset.strip()])

        docs_config = config.get("docs_config")
        queries_config = config.get("queries_config")
        qrels_config = config.get("qrels_config")
        if isinstance(docs_config, str) and docs_config.strip():
            out.extend(["--docs-config", docs_config.strip()])
        if isinstance(queries_config, str) and queries_config.strip():
            out.extend(["--queries-config", queries_config.strip()])
        if isinstance(qrels_config, str) and qrels_config.strip():
            out.extend(["--qrels-config", qrels_config.strip()])

        max_documents = config.get("max_documents")
        if isinstance(max_documents, int) and max_documents > 0:
            out.extend(["--max-documents", str(max_documents)])

        max_queries = config.get("max_queries")
        if isinstance(max_queries, int) and max_queries > 0:
            out.extend(["--max-queries", str(max_queries)])

        max_chars = config.get("max_chars_per_doc")
        if isinstance(max_chars, int) and max_chars > 0:
            out.extend(["--max-chars-per-doc", str(max_chars)])

        return out
