from __future__ import annotations

from typing import Dict, Type

from src.datasets.base import BaseDatasetProcessor
from src.datasets.beir import BEIRDatasetProcessor
from src.datasets.dummy import DummyDatasetProcessor
from src.datasets.msmarco import MSMarcoDatasetProcessor


class DatasetFactory:
    _registry: Dict[str, Type[BaseDatasetProcessor]] = {
        "default": DummyDatasetProcessor,
        "msmarco-docs": MSMarcoDatasetProcessor,
        "msmarco": MSMarcoDatasetProcessor,
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
            return BEIRDatasetProcessor(dataset_name=dataset_name)

        if normalized_name in cls._registry:
            return cls._registry[normalized_name]()

        available = sorted(list(cls._registry.keys()) + ["beir/<dataset_name>"])
        raise ValueError(f"Unknown dataset processor: '{name}'. Available: {', '.join(available)}")