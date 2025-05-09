# -----------------------------------------------------------------------------
# src/emafusion/judge.py
# -----------------------------------------------------------------------------
"""Answer quality judges – ensemble or single‑LLM (paper§5)."""
from __future__ import annotations

import os
from typing import Protocol

try:
    import openai
except ImportError:  # optional, users might rely on local judge
    openai = None  # type: ignore


class Judge(Protocol):
    """Return confidence ∈ [0,1] for <prompt, answer>."""

    def score(self, prompt: str, draft: str) -> float:  # pragma: no cover – interface only
        ...


class LLMJudge:
    """Default implementation using OpenAI chat‑completion log‑probs."""

    def __init__(self, model: str = "gpt-3.5-turbo-0125"):
        if openai is None:
            raise RuntimeError("openai‑python not installed; cannot use LLMJudge.")
        self.model = model
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if openai.api_key is None:
            raise EnvironmentError("OPENAI_API_KEY not set.")

    def score(self, prompt: str, draft: str) -> float:
        # Very light‑weight: ask model if answer is correct on 3‑level scale → map to [0,1]
        sys_msg = (
            "You are an impartial grader. Answer only with a number: 1 (correct), 0.5 (uncertain), 0 (incorrect)."
        )
        user_msg = f"Question: {prompt}\nModel answer: {draft}\nScore:"
        resp = openai.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}],
            max_tokens=1,
            temperature=0.0,
        )
        content = resp.choices[0].message.content.strip()
        try:
            return float(content)
        except ValueError:
            return 0.0