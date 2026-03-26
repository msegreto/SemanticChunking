from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Type

try:
    import pyterrier as pt
except ImportError:  # pragma: no cover
    pt = None

from src.splitting.contextualizer import ContextualizerSplitter
from src.splitting.sentence import SentenceSplitter


_PtTransformerBase = pt.Transformer if pt is not None else object


class SplitTransformer(_PtTransformerBase):
    """
    Offline, cache-aware splitter that can be used as the corpus-side
    transformation stage between a normalized dataset and downstream chunking.
    """

    SCHEMA_VERSION = "1.0"
    _REGISTRY: dict[str, Type[Any]] = {
        "sentence": SentenceSplitter,
        "contextualizer": ContextualizerSplitter,
    }

    def __init__(
        self,
        *,
        splitter: Any,
        split_config: dict[str, Any],
        dataset_name: str,
        run_name: str,
        allow_reuse: bool = True,
        force_rebuild: bool = False,
    ) -> None:
        self.splitter = splitter
        self.split_config = dict(split_config)
        self.dataset_name = str(dataset_name).strip()
        self.run_name = str(run_name).strip() or "default_run"
        self.allow_reuse = bool(allow_reuse)
        self.force_rebuild = bool(force_rebuild)
        self._components: dict[str, Any] | None = None

    @classmethod
    def from_config(
        cls,
        *,
        split_config: dict[str, Any],
        dataset_name: str,
        run_name: str,
        allow_reuse: bool = True,
        force_rebuild: bool = False,
    ) -> "SplitTransformer":
        split_type = str(split_config.get("type", "sentence")).strip().lower() or "sentence"
        try:
            splitter_cls = cls._REGISTRY[split_type]
        except KeyError as exc:
            available = ", ".join(sorted(cls._REGISTRY.keys()))
            raise ValueError(f"Unknown splitter: '{split_type}'. Available: {available}") from exc

        return cls(
            splitter=splitter_cls(),
            split_config=split_config,
            dataset_name=dataset_name,
            run_name=run_name,
            allow_reuse=allow_reuse,
            force_rebuild=force_rebuild,
        )

    def transform(self, inp: Any) -> dict[str, Any]:
        if not self.force_rebuild and self.allow_reuse:
            reusable = self.try_load_reusable_output()
            if reusable is not None:
                return reusable
        return self.build(inp)

    def build(self, inp: Any) -> dict[str, Any]:
        output_path = self.resolve_output_path()
        metadata_path = self.resolve_metadata_path()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        components = self.get_components()
        total_docs = 0
        total_units = 0
        next_unitno = 0
        output_path.write_text("", encoding="utf-8")

        with output_path.open("a", encoding="utf-8") as f_out:
            for document in self.iter_documents(inp):
                docno = str(document.get("docno") or "").strip()
                if not docno:
                    continue
                text = self.resolve_text(document)
                units, next_unitno = self.splitter.split_document_streaming(
                    docno=docno,
                    text=text,
                    unitno_start=next_unitno,
                    nlp=components["nlp"],
                    max_chars_per_batch=int(components["max_chars_per_batch"]),
                )
                for unit in units:
                    f_out.write(json.dumps(unit, ensure_ascii=False) + "\n")
                total_docs += 1
                total_units += len(units)

        metadata = {
            "split_schema_version": self.SCHEMA_VERSION,
            "dataset_name": self.dataset_name,
            "run_name": self.run_name,
            "splitter_type": self.splitter_name,
            "split_config": self._stable_split_config(),
            "output_path": str(output_path.resolve()),
            "num_documents": total_docs,
            "num_units": total_units,
            "spacy_model": components.get("model_name"),
        }
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

        return {
            "split_units_path": str(output_path.resolve()),
            "split_units": self.load_split_units(output_path),
            "metadata": metadata,
        }

    def get_components(self) -> dict[str, Any]:
        if self._components is None:
            self._components = self.splitter.build_streaming_components(self.split_config)
        return self._components

    def transform_document(
        self,
        *,
        document: dict[str, Any],
        unitno_start: int,
    ) -> tuple[list[dict[str, Any]], int]:
        docno = str(document.get("docno") or "").strip()
        if not docno:
            return [], unitno_start
        text = self.resolve_text(document)
        components = self.get_components()
        return self.splitter.split_document_streaming(
            docno=docno,
            text=text,
            unitno_start=unitno_start,
            nlp=components["nlp"],
            max_chars_per_batch=int(components["max_chars_per_batch"]),
        )

    def try_load_reusable_output(self) -> dict[str, Any] | None:
        output_path = self.resolve_output_path()
        metadata_path = self.resolve_metadata_path()
        if not output_path.exists() or not metadata_path.exists():
            return None
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if not self._metadata_matches(metadata):
            return None
        return {
            "split_units_path": str(output_path.resolve()),
            "split_units": self.load_split_units(output_path),
            "metadata": metadata,
        }

    def resolve_output_path(self) -> Path:
        explicit = self.splitter.resolve_output_path(self.split_config)
        if explicit is not None:
            return explicit
        return (
            Path("data/splits")
            / self._slugify(self.dataset_name)
            / f"{self._slugify(self.run_name)}--{self.splitter_name}.jsonl"
        )

    def resolve_metadata_path(self) -> Path:
        output_path = self.resolve_output_path()
        return output_path.with_suffix(output_path.suffix + ".meta.json")

    @property
    def splitter_name(self) -> str:
        return self.splitter.__class__.__name__.replace("Splitter", "").lower() or "splitter"

    def iter_documents(self, inp: Any) -> Iterable[dict[str, Any]]:
        if hasattr(inp, "get_corpus_iter"):
            yield from inp.get_corpus_iter(verbose=True)
            return

        if isinstance(inp, dict):
            corpus = inp.get("corpus")
            if isinstance(corpus, list):
                for row in corpus:
                    if isinstance(row, dict):
                        yield row
                return

        if isinstance(inp, list):
            for row in inp:
                if isinstance(row, dict):
                    yield row
            return

        raise TypeError(
            "SplitTransformer input must expose get_corpus_iter(), or be a payload with 'corpus', or be a list of documents."
        )

    @staticmethod
    def resolve_text(document: dict[str, Any]) -> str:
        for key in ("text", "contents", "body", "document"):
            value = document.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return ""

    @staticmethod
    def load_split_units(path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    def _metadata_matches(self, metadata: Any) -> bool:
        if not isinstance(metadata, dict):
            return False
        return (
            metadata.get("split_schema_version") == self.SCHEMA_VERSION
            and metadata.get("dataset_name") == self.dataset_name
            and metadata.get("run_name") == self.run_name
            and metadata.get("splitter_type") == self.splitter_name
            and metadata.get("split_config") == self._stable_split_config()
        )

    def _stable_split_config(self) -> dict[str, Any]:
        tracked = dict(self.split_config)
        tracked.pop("output_path", None)
        tracked.pop("save_output", None)
        return tracked

    @staticmethod
    def _slugify(value: str) -> str:
        text = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value.strip())
        return text.strip("._") or "item"
