from __future__ import annotations

from typing import Dict, Type

from src.datasets.base import BaseDatasetProcessor
from src.datasets.beir import BEIRDatasetProcessor
from src.datasets.msmarco import MSMarcoDatasetProcessor
from src.datasets.scripted import QasperScriptedProcessor, TechQAScriptedProcessor


class DatasetFactory:
    _registry: Dict[str, Type[BaseDatasetProcessor]] = {
        # `msmarco` and `msmarco-docs` both refer to the official MS MARCO
        # document ranking collection. `beir/msmarco` is handled separately
        # by the generic BEIR processor below.
        "msmarco-docs": MSMarcoDatasetProcessor,
        "msmarco": MSMarcoDatasetProcessor,
    }
    _beir_overrides: Dict[str, Type[BaseDatasetProcessor]] = {
        # These datasets are ingested via dedicated converters that produce
        # project-normalized artifacts directly under data/normalized/.
        "qasper": QasperScriptedProcessor,
        "techqa": TechQAScriptedProcessor,
    }

    @classmethod
    def register(cls, name: str, processor_cls: Type[BaseDatasetProcessor]) -> None:
        cls._registry[name] = processor_cls

    @classmethod
    def create(cls, name: str) -> BaseDatasetProcessor:
        normalized_name = name.strip()

        if normalized_name.startswith("beir/"):
            dataset_name = normalized_name.split("/", 1)[1]
            if not dataset_name:
                raise ValueError("BEIR dataset name missing. Expected format: 'beir/<dataset_name>'")
            override_cls = cls._beir_overrides.get(dataset_name)
            if override_cls is not None:
                return override_cls()
            return BEIRDatasetProcessor(dataset_name=dataset_name)

        if normalized_name in cls._registry:
            return cls._registry[normalized_name]()

        available = sorted(list(cls._registry.keys()) + ["beir/<dataset_name>"])
        raise ValueError(f"Unknown dataset processor: '{name}'. Available: {', '.join(available)}")
