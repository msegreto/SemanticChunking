from __future__ import annotations

from typing import Dict, Type

from src.routing.base import BaseRouter
from src.routing.dummy import DummyRouter


class RouterFactory:
    _registry: Dict[str, Type[BaseRouter]] = {
        "default": DummyRouter,
        "moc": DummyRouter,
    }

    @classmethod
    def register(cls, name: str, router_cls: Type[BaseRouter]) -> None:
        cls._registry[name] = router_cls

    @classmethod
    def create(cls, name: str) -> BaseRouter:
        if name not in cls._registry:
            available = ", ".join(sorted(cls._registry.keys()))
            raise ValueError(f"Unknown router: '{name}'. Available: {available}")
        return cls._registry[name]()