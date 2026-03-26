from __future__ import annotations

DEFAULT_CONTEXTUALIZER_PROMPT = """You are a minimal-edit contextualizer.
Task: keep the sentence text unchanged except for reference resolution.
Rules:
1) Replace pronouns or ambiguous references with explicit entity names when inferable from context.
2) Do NOT paraphrase, summarize, reorder, add, or remove information.
3) Preserve wording, tense, punctuation, and style as much as possible.
4) Keep exactly one sentence as output.
5) If no safe reference resolution is possible, return the original sentence unchanged.
6) Return only the final sentence.

Sentence:
{text}

Final sentence:"""
