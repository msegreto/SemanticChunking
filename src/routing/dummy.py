from __future__ import annotations

from typing import Any

from src.routing.base import BaseRouter


class DummyRouter(BaseRouter):
    def route(self, split_output: Any, config: dict) -> Any:
        print(f"[ROUTER] route() with config={config}")
        output = dict(split_output)
        output["router_info"] = {"router_used": True, "router_name": config.get("name", "default")}
        return output