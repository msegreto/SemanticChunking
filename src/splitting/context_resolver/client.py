from __future__ import annotations

import json
import time
from typing import Any
from urllib import error, request


class HFContextualizerClient:
    def __init__(
        self,
        *,
        endpoint_url: str | None,
        api_token: str | None = None,
        timeout_seconds: float = 45.0,
        max_retries: int = 2,
        retry_backoff_seconds: float = 1.0,
        generation_params: dict[str, Any] | None = None,
    ) -> None:
        self.endpoint_url = (endpoint_url or "").strip()
        self.api_token = (api_token or "").strip()
        self.timeout_seconds = float(max(1.0, timeout_seconds))
        self.max_retries = int(max(0, max_retries))
        self.retry_backoff_seconds = float(max(0.0, retry_backoff_seconds))
        self.generation_params = dict(generation_params or {})

    def is_configured(self) -> bool:
        return bool(self.endpoint_url)

    def contextualize_text(self, text: str) -> str:
        payload: dict[str, Any] = {"inputs": text}
        if self.generation_params:
            payload["parameters"] = self.generation_params
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"

        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                req = request.Request(self.endpoint_url, data=body, headers=headers, method="POST")
                with request.urlopen(req, timeout=self.timeout_seconds) as resp:
                    raw = resp.read().decode("utf-8")
                parsed = json.loads(raw)
                return self._extract_generated_text(parsed, fallback=text)
            except (TimeoutError, error.URLError, error.HTTPError, json.JSONDecodeError) as exc:
                last_err = exc
                if attempt >= self.max_retries:
                    break
                if self.retry_backoff_seconds > 0:
                    time.sleep(self.retry_backoff_seconds * (attempt + 1))

        if last_err is None:
            return text
        raise RuntimeError(f"HF contextualizer request failed: {last_err}") from last_err

    @staticmethod
    def _extract_generated_text(payload: Any, *, fallback: str) -> str:
        # HF Inference commonly returns:
        # - [{"generated_text": "..."}]
        # - {"generated_text": "..."}
        # - {"error": "..."} on failure
        if isinstance(payload, list) and payload:
            first = payload[0]
            if isinstance(first, dict) and isinstance(first.get("generated_text"), str):
                text = first["generated_text"].strip()
                return text if text else fallback
            if isinstance(first, str) and first.strip():
                return first.strip()

        if isinstance(payload, dict):
            if isinstance(payload.get("generated_text"), str):
                text = payload["generated_text"].strip()
                return text if text else fallback
            if isinstance(payload.get("output_text"), str):
                text = payload["output_text"].strip()
                return text if text else fallback

        return fallback
