from __future__ import annotations

from copy import deepcopy
from typing import Any

from src.splitting.context_resolver.client import HFContextualizerClient
from src.splitting.context_resolver.prompts import DEFAULT_CONTEXTUALIZER_PROMPT


class Contextualizer:
    def __init__(
        self,
        *,
        enabled: bool,
        client: HFContextualizerClient | None,
        prompt_template: str | None = None,
        max_input_chars: int = 3000,
        fail_on_error: bool = False,
    ) -> None:
        self.enabled = bool(enabled)
        self.client = client
        self.prompt_template = str(prompt_template or DEFAULT_CONTEXTUALIZER_PROMPT)
        self.max_input_chars = int(max(1, max_input_chars))
        self.fail_on_error = bool(fail_on_error)

    def contextualize_units(self, units: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not self.enabled or not units:
            return units
        if self.client is None or not self.client.is_configured():
            return units

        output: list[dict[str, Any]] = []
        for unit in units:
            raw_text = str(unit.get("text", "")).strip()
            if not raw_text:
                output.append(unit)
                continue

            rewritten = raw_text
            try:
                prompt = self._build_prompt(raw_text)
                model_output = self.client.contextualize_text(prompt)
                rewritten = self._extract_rewritten_sentence(model_output=model_output, fallback=raw_text)
            except Exception:
                if self.fail_on_error:
                    raise

            updated = deepcopy(unit)
            updated["text"] = rewritten
            output.append(updated)

        return output

    def _build_prompt(self, text: str) -> str:
        safe_text = text[: self.max_input_chars]
        template = self.prompt_template if "{text}" in self.prompt_template else f"{self.prompt_template}\n{{text}}"
        return template.format(text=safe_text)

    @staticmethod
    def _extract_rewritten_sentence(*, model_output: str, fallback: str) -> str:
        if not isinstance(model_output, str):
            return fallback
        candidate = model_output.strip()
        if not candidate:
            return fallback

        # If the model echoes the prompt, keep only tail after the last known marker.
        markers = ["Rewritten sentence:", "Output:", "Answer:"]
        lowered = candidate.lower()
        cut_idx = -1
        for marker in markers:
            idx = lowered.rfind(marker.lower())
            if idx >= 0:
                cut_idx = max(cut_idx, idx + len(marker))
        if cut_idx >= 0:
            tail = candidate[cut_idx:].strip()
            if tail:
                candidate = tail

        first_line = candidate.splitlines()[0].strip()
        return first_line if first_line else fallback
