from __future__ import annotations

from typing import Any, Type

try:
    import pyterrier as pt
except ImportError:  # pragma: no cover
    pt = None

from src.chunking.fixed import FixedChunker
from src.chunking.semantic_breakpoint import SemanticBreakpointChunker
from src.chunking.semantic_clustering import SemanticClusteringChunker


_PtTransformerBase = pt.Transformer if pt is not None else object


class ChunkTransformer(_PtTransformerBase):
    _REGISTRY: dict[str, Type[Any]] = {
        "fixed": FixedChunker,
        "semantic_breakpoint": SemanticBreakpointChunker,
        "semantic_clustering": SemanticClusteringChunker,
    }

    def __init__(
        self,
        *,
        chunker: Any,
        chunking_config: dict[str, Any],
        allow_reuse: bool = True,
        force_rebuild: bool = False,
    ) -> None:
        self.chunker = chunker
        self.chunking_config = dict(chunking_config)
        self.allow_reuse = bool(allow_reuse)
        self.force_rebuild = bool(force_rebuild)
        self.validated_config = self.chunker.validate_config(self.chunking_config)

    @classmethod
    def from_config(
        cls,
        *,
        chunking_config: dict[str, Any],
        allow_reuse: bool = True,
        force_rebuild: bool = False,
    ) -> "ChunkTransformer":
        chunking_type = str(chunking_config.get("type", "")).strip().lower()
        if not chunking_type:
            raise ValueError("Missing required config: chunking.type")
        try:
            chunker_cls = cls._REGISTRY[chunking_type]
        except KeyError as exc:
            available = ", ".join(sorted(cls._REGISTRY.keys()))
            raise ValueError(f"Unknown chunker: '{chunking_type}'. Available: {available}") from exc

        return cls(
            chunker=chunker_cls(),
            chunking_config=chunking_config,
            allow_reuse=allow_reuse,
            force_rebuild=force_rebuild,
        )

    def transform(self, inp: Any) -> dict[str, Any]:
        return self.build(inp)

    def build(self, inp: Any) -> dict[str, Any]:
        if not self.force_rebuild and self.allow_reuse:
            return self.chunker.chunk(inp, self.validated_config)

        prepared = self.chunker.prepare_chunking_inputs(inp, self.validated_config)
        chunk_output = self.chunker.compute_chunk_output(prepared)
        self.chunker.save_output(chunk_output, prepared)
        return chunk_output

    def resolve_output_dir(self) -> str:
        run_dir = self.chunker.build_run_dir(
            dataset=self.validated_config["dataset"],
            yaml_name=self.validated_config["yaml_name"],
        )
        return str(run_dir)
