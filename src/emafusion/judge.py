# -----------------------------------------------------------------------------
# src/emafusion/judge.py
# -----------------------------------------------------------------------------
"""Answer quality judges – ensemble or single‑LLM (paper§5)."""
"""Confidence‑scoring judges for draft answers."""
from __future__ import annotations

import asyncio
import os
from typing import Protocol

import openai  # type: ignore


class Judge(Protocol):
    """Generic scoring interface."""

    async def score(self, prompt: str, draft: str) -> float:  # noqa: D401 simple docstring OK
        """Return confidence ∈ [0,1]."""


class LLMJudge:
    """LLM‑based grader. One API call → returns a float confidence."""

    def __init__(self, model: str):
        openai.api_key = os.getenv("OPENAI_API_KEY", "")
        self._model = model

    async def score(self, prompt: str, draft: str) -> float:  # noqa: D401
        completion = await openai.ChatCompletion.acreate(
            model=self._model,
            temperature=0,
            max_tokens=4,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Act as a strict grader. Return **only** a float between 0 and 1 "
                        "representing how correct and complete the answer is."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Prompt:\n{prompt}\n\nAnswer:\n{draft}\n\nScore:",
                },
            ],
        )
        content = completion.choices[0].message.content.strip()
        try:
            return max(0.0, min(1.0, float(content)))
        except ValueError:
            return 0.0