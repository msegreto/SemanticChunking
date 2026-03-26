from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from src.splitting.context_resolver.client import HFContextualizerClient
from src.splitting.context_resolver.resolver import Contextualizer
from src.splitting.sentence import SentenceSplitter


class ContextualizerSplitter:
    MAX_CHARS_PER_BATCH = SentenceSplitter.MAX_CHARS_PER_BATCH

    @staticmethod
    def resolve_output_path(config: dict) -> Path | None:
        output_path = config.get("output_path")
        return Path(output_path) if output_path else None

    def __init__(self):
        self._sentence_splitter = SentenceSplitter()

    def build_streaming_components(self, config: dict) -> dict[str, Any]:
        sentence_components = self._sentence_splitter.build_streaming_components(config)
        resolver_cfg = self._resolve_config(config)
        enabled = bool(resolver_cfg.get("enabled", False))
        backend = str(resolver_cfg.get("backend", "hf_remote")).strip().lower()
        if enabled and backend != "hf_remote":
            raise ValueError("Contextualizer supports only split.context_resolver.backend='hf_remote'.")
        client = self._build_client(resolver_cfg)
        if enabled and (client is None or not client.is_configured()):
            raise ValueError(
                "Contextualizer enabled but endpoint is not configured. "
                "Set split.context_resolver.endpoint_url or endpoint_url_env."
            )
        contextualizer = Contextualizer(
            enabled=enabled,
            client=client,
            prompt_template=str(resolver_cfg.get("prompt_template", "")).strip() or None,
            max_input_chars=int(resolver_cfg.get("max_input_chars", 3000)),
            fail_on_error=bool(resolver_cfg.get("fail_on_error", False)),
        )
        return {
            "model_name": sentence_components["model_name"],
            "nlp": {
                "sentence_nlp": sentence_components["nlp"],
                "contextualizer": contextualizer,
            },
            "max_chars_per_batch": sentence_components["max_chars_per_batch"],
        }

    def split_document_streaming(
        self,
        *,
        docno: str,
        text: str,
        unitno_start: int,
        nlp: Any,
        max_chars_per_batch: int,
    ) -> tuple[list[dict[str, Any]], int]:
        if not isinstance(nlp, dict):
            raise TypeError("Contextualizer splitter expects 'nlp' to be a dict with sentence_nlp/contextualizer.")

        sentence_nlp = nlp.get("sentence_nlp")
        contextualizer = nlp.get("contextualizer")
        if sentence_nlp is None:
            raise ValueError("Missing sentence_nlp in contextualizer splitter components.")

        units, next_unit_id = self._sentence_splitter.split_document_streaming(
            docno=docno,
            text=text,
            unitno_start=unitno_start,
            nlp=sentence_nlp,
            max_chars_per_batch=max_chars_per_batch,
        )
        if isinstance(contextualizer, Contextualizer):
            units = contextualizer.contextualize_units(units)
        return units, next_unit_id

    @staticmethod
    def _resolve_config(config: dict[str, Any]) -> dict[str, Any]:
        # Keep all contextualizer settings under split.context_resolver to keep experiment files explicit.
        raw = config.get("context_resolver", {})
        if not isinstance(raw, dict):
            raw = {}
        return dict(raw)

    @staticmethod
    def _build_client(cfg: dict[str, Any]) -> HFContextualizerClient | None:
        endpoint_url = str(cfg.get("endpoint_url", "")).strip()
        endpoint_env = str(cfg.get("endpoint_url_env", "")).strip()
        if not endpoint_url and endpoint_env:
            endpoint_url = str(os.getenv(endpoint_env, "")).strip()

        token = str(cfg.get("api_token", "")).strip()
        token_env = str(cfg.get("api_token_env", "")).strip()
        if not token and token_env:
            token = str(os.getenv(token_env, "")).strip()

        if not endpoint_url:
            return None

        timeout_seconds = float(cfg.get("timeout_seconds", 45))
        max_retries = int(cfg.get("max_retries", 2))
        retry_backoff_seconds = float(cfg.get("retry_backoff_seconds", 1.0))
        generation_params = {
            "temperature": float(cfg.get("temperature", 0.0)),
            "do_sample": bool(cfg.get("do_sample", False)),
            "return_full_text": bool(cfg.get("return_full_text", False)),
        }
        if "max_new_tokens" in cfg:
            generation_params["max_new_tokens"] = int(cfg.get("max_new_tokens", 64))
        if "top_p" in cfg:
            generation_params["top_p"] = float(cfg.get("top_p", 1.0))
        return HFContextualizerClient(
            endpoint_url=endpoint_url,
            api_token=token,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
            generation_params=generation_params,
        )
