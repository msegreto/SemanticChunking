from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, Optional, Tuple

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:  # pragma: no cover
    AutoModelForCausalLM = None
    AutoTokenizer = None


class PerplexityScorer:
    """
    Scorer abstraction for intrinsic metrics.

    Supported backends:
    - lexical
    - hf_causal_lm

    For hf_causal_lm:
    - perplexity(q) is computed with target-only loss on q
    - conditional_perplexity(q | d) is computed on [d + q],
      masking context tokens from the loss
    """

    def __init__(
        self,
        backend: str = "lexical",
        model_name: Optional[str] = None,
        device: str = "auto",
        max_length: int = 1024,
        trust_remote_code: bool = False,
    ) -> None:
        self.backend = backend
        self.model_name = model_name
        self.device = device
        self.max_length = int(max_length)
        self.trust_remote_code = trust_remote_code
        self.backend_name = backend

        self._model = None
        self._tokenizer = None
        self._resolved_device = None

        self._ppl_cache: Dict[str, float] = {}
        self._cond_ppl_cache: Dict[Tuple[str, str], float] = {}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "PerplexityScorer":
        backend = str(config.get("backend", "lexical")).strip().lower()
        model_name = config.get("name")
        device = str(config.get("device", "auto")).strip().lower()
        max_length = int(config.get("max_length", 1024))
        trust_remote_code = bool(config.get("trust_remote_code", False))

        return cls(
            backend=backend,
            model_name=model_name,
            device=device,
            max_length=max_length,
            trust_remote_code=trust_remote_code,
        )

    def perplexity(self, text: str) -> float:
        if self.backend == "lexical":
            return self._lexical_perplexity(text)

        if self.backend == "hf_causal_lm":
            if text in self._ppl_cache:
                return self._ppl_cache[text]

            value = self._hf_perplexity(
                target_text=text,
                context_text=None,
            )
            self._ppl_cache[text] = value
            return value

        raise ValueError(f"Unknown scorer backend: {self.backend}")

    def conditional_perplexity(self, target_text: str, context_text: str) -> float:
        if self.backend == "lexical":
            return self._lexical_conditional_perplexity(target_text, context_text)

        if self.backend == "hf_causal_lm":
            key = (target_text, context_text)
            if key in self._cond_ppl_cache:
                return self._cond_ppl_cache[key]

            value = self._hf_perplexity(
                target_text=target_text,
                context_text=context_text,
            )
            self._cond_ppl_cache[key] = value
            return value

        raise ValueError(f"Unknown scorer backend: {self.backend}")

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def _lexical_perplexity(self, text: str) -> float:
        tokens = self._tokenize(text)
        if not tokens:
            return 1.0

        counts = Counter(tokens)
        total = len(tokens)

        entropy = 0.0
        for count in counts.values():
            p = count / total
            entropy -= p * math.log(p)

        return math.exp(entropy)

    def _lexical_conditional_perplexity(self, target_text: str, context_text: str) -> float:
        target_tokens = self._tokenize(target_text)
        context_tokens = self._tokenize(context_text)

        if not target_tokens:
            return 1.0

        base = self._lexical_perplexity(target_text)

        if not context_tokens:
            return base

        target_set = set(target_tokens)
        context_set = set(context_tokens)

        overlap = len(target_set & context_set)
        denom = len(target_set) if target_set else 1
        overlap_ratio = overlap / denom

        adjusted = base * (1.0 - 0.5 * overlap_ratio)
        return max(1e-6, adjusted)

    def _hf_perplexity(self, target_text: str, context_text: Optional[str]) -> float:
        self._ensure_hf_backend_ready()

        target_text = target_text or ""
        context_text = context_text or ""

        target_ids = self._encode_text(target_text)
        context_ids = self._encode_text(context_text) if context_text else []

        if not target_ids:
            return 1.0

        input_ids_list, context_len = self._build_sequence(
            context_ids=context_ids,
            target_ids=target_ids,
        )

        if len(input_ids_list) < 2:
            return 1.0

        input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=self._resolved_device)
        attention_mask = torch.ones_like(input_ids)

        labels = input_ids.clone()
        if context_len > 0:
            labels[:, :context_len] = -100

        with torch.no_grad():
            outputs = self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss

        if loss is None:
            return 1.0

        return float(torch.exp(loss).detach().cpu().item())

    def _build_sequence(self, context_ids: list[int], target_ids: list[int]) -> Tuple[list[int], int]:
        """
        Preserve the whole target whenever possible.
        If needed, truncate context first.
        If target alone exceeds max_length, keep the last max_length tokens.
        """
        if len(target_ids) >= self.max_length:
            truncated_target = target_ids[-self.max_length :]
            return truncated_target, 0

        remaining = self.max_length - len(target_ids)
        truncated_context = context_ids[-remaining:] if remaining > 0 else []

        full_ids = truncated_context + target_ids
        context_len = len(truncated_context)
        return full_ids, context_len

    def _encode_text(self, text: str) -> list[int]:
        if not text:
            return []

        encoded = self._tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return list(encoded["input_ids"])

    def _ensure_hf_backend_ready(self) -> None:
        if self.backend != "hf_causal_lm":
            return

        if torch is None or AutoTokenizer is None or AutoModelForCausalLM is None:
            raise ImportError(
                "hf_causal_lm backend requires torch and transformers to be installed."
            )

        if self._model is not None and self._tokenizer is not None:
            return

        if not self.model_name:
            raise ValueError(
                "intrinsic_model.name must be provided when backend='hf_causal_lm'."
            )

        self._resolved_device = self._resolve_device(self.device)

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
        )

        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
        )
        self._model.to(self._resolved_device)
        self._model.eval()

    def _resolve_device(self, requested: str):
        if requested == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")

        if requested == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("Requested device 'cuda' but CUDA is not available.")
            return torch.device("cuda")

        if requested == "mps":
            if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                raise ValueError("Requested device 'mps' but MPS is not available.")
            return torch.device("mps")

        if requested == "cpu":
            return torch.device("cpu")

        raise ValueError(f"Unsupported device setting: {requested}")