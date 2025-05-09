# -----------------------------------------------------------------------------
# src/emafusion/cascade.py
# -----------------------------------------------------------------------------
"""Cheapest‑to‑priciest answer cascade (paper §5)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any

from .judge import Judge


@dataclass
class _ModelClient:
    name: str
    price_prompt: float
    price_completion: float
    max_tokens: int
    provider: str
    _client: Any  # provider‑specific handle

    def generate(self, prompt: str, **kw) -> str:
        if self.provider == "openai":
            return self._openai(prompt, **kw)
        elif self.provider == "ollama":
            return self._ollama(prompt, **kw)
        raise NotImplementedError(self.provider)

    # --- provider glue --------------------------------------------------------
    def _openai(self, prompt: str, **kw) -> str:
        import openai  # local import to keep optional

        resp = openai.chat.completions.create(
            model=self.name,
            messages=[{"role": "user", "content": prompt}],
            temperature=kw.get("temperature", 0.2),
            max_tokens=min(self.max_tokens, kw.get("max_tokens", 512)),
        )
        return resp.choices[0].message.content

    def _ollama(self, prompt: str, **kw) -> str:
        import ollama  # type: ignore – optional

        resp = ollama.chat(model=self.name, messages=[{"role": "user", "content": prompt}])
        return resp["message"]["content"]


class Cascade:
    """Iteratively call model list until judge confidence ≥ τ."""

    def __init__(self, models: List[_ModelClient], judge: Judge, tau: float = 0.6):
        self.models = models
        self.judge = judge
        self.tau = tau

    @classmethod
    def from_config(cls, cfg: "emafusion.config.Settings") -> "Cascade":  # noqa: F821
        from .config import ModelCfg  # local import to avoid cycle

        clients: List[_ModelClient] = []
        for m in cfg.models:
            if m.provider == "openai":
                import openai

                openai.api_key = os.getenv("OPENAI_API_KEY")
                client = _ModelClient(
                    name=m.name,
                    provider="openai",
                    price_prompt=m.price_prompt,
                    price_completion=m.price_completion,
                    max_tokens=m.max_tokens,
                    _client=openai,
                )
            elif m.provider == "ollama":
                import ollama  # type: ignore

                client = _ModelClient(
                    name=m.name,
                    provider="ollama",
                    price_prompt=m.price_prompt,
                    price_completion=m.price_completion,
                    max_tokens=m.max_tokens,
                    _client=ollama,
                )
            else:
                raise ValueError(f"Unknown provider {m.provider}")
            clients.append(client)
        judge = LLMJudge(cfg.judge_model)  # default implementation
        return cls(clients, judge, tau=cfg.thresholds.get("confidence", 0.6))

    # ---------------------------------------------------------------------
    def answer(self, prompt: str, stop_at_first: bool = True) -> str:
        drafts: List[tuple[str, float]] = []
        for client in self.models:
            draft = client.generate(prompt)
            score = self.judge.score(prompt, draft)
            drafts.append((draft, score))
            if score >= self.tau and stop_at_first:
                return draft
        # If nothing passed threshold pick winner by fusion
        from .fusion import fuse_answers

        return fuse_answers(prompt, drafts)