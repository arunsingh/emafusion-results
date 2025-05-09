# -----------------------------------------------------------------------------
# src/emafusion/cascade.py
# -----------------------------------------------------------------------------
"""Cheapest‑to‑priciest answer cascade (paper§5)."""
"""Confidence‑gated *cheap→expensive* model cascade."""
from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import List, Protocol

from .judge import LLMJudge


class _ModelClient(Protocol):
    """Simplified model client interface expected by the cascade."""

    name: str

    async def generate(self, prompt: str) -> str:  # noqa: D401 – short doc
        ...


@dataclass(slots=True)
class Cascade:
    """Implements the EMAFusion escalation logic."""

    clients: List[_ModelClient]
    judge: LLMJudge
    tau: float = 0.6

    # ------------------------------------------------------------------
    # Runtime API
    # ------------------------------------------------------------------
    async def answer(self, prompt: str) -> str:  # noqa: D401
        draft: str | None = None
        for client in self.clients:
            draft = await client.generate(prompt)
            conf = await self.judge.score(prompt, draft)
            if conf >= self.tau:
                return draft
        # Fallback: return last draft (low‑confidence) if nothing exceeded τ
        return draft or ""

    # ------------------------------------------------------------------
    # Factory from Settings
    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, cfg: "emafusion.config.Settings") -> "Cascade":  # noqa: F821
        """Instantiate clients + judge from :class:`~emafusion.config.Settings`."""

        clients: List[_ModelClient] = []

        # Local import to avoid heavy deps if not needed
        import openai  # type: ignore

        # ----------------------------------------------
        # Dynamically build wrapper objects per provider
        # ----------------------------------------------
        for m in cfg.models:
            if m.provider == "openai":

                openai.api_key = os.getenv("OPENAI_API_KEY", "")

                class _OpenAIClient:  # noqa: D401 – inner class OK
                    def __init__(self, name: str, engine: str):
                        self.name = name
                        self._engine = engine

                    async def generate(self, prompt: str) -> str:
                        resp = await openai.ChatCompletion.acreate(
                            model=self._engine,
                            messages=[{"role": "user", "content": prompt}],
                        )
                        return resp.choices[0].message.content

                clients.append(_OpenAIClient(m.name, m.engine))

            else:
                raise ValueError(f"Unknown provider '{m.provider}' in config")

        judge = LLMJudge(cfg.judge_model)
        return cls(clients, judge, tau=cfg.thresholds.get("confidence", 0.6))

