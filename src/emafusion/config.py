# -----------------------------------------------------------------------------
# src/emafusion/config.py
# -----------------------------------------------------------------------------

"""Configuration helpers.

YAML schema::
  models:
    - name: llama3-8b
      provider: ollama
      price_prompt: 0.0
      price_completion: 0.0
      max_tokens: 4096
    - name: gpt-4o
      provider: openai
      price_prompt: 0.01
      price_completion: 0.03
  taxonomy_index: artifacts/taxonomy.index
  learned_ckpt: artifacts/router.pkl
  judge_model: gpt-3.5-turbo
  thresholds:
    confidence: 0.6
"""
"""Configuration dataclasses and helpers."""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field
from typing import Dict, List

import yaml  # type: ignore


@dataclass(slots=True)
class ModelCfg:
    """Per‑model runtime metadata."""

    name: str  # Friendly label, e.g. "llama3-8b-local"
    provider: str  # "openai", "ollama", "together", etc.
    engine: str  # Underlying model identifier
    cost: float | None = None  # $ /1k tokens (optional)


@dataclass(slots=True)
class Settings:
    """Cascade + judge configuration loaded from YAML/JSON."""

    models: List[ModelCfg]
    judge_model: str = "gpt-3.5-turbo"
    thresholds: Dict[str, float] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: str | pathlib.Path) -> "Settings":  # noqa: D401 – simple docstring
        """Read **path** (YAML/JSON) into a :class:`Settings` instance."""

        path = pathlib.Path(path)
        if not path.exists():
            raise FileNotFoundError(path)

        if path.suffix in {".yaml", ".yml"}:
            data = yaml.safe_load(path.read_text())
        else:
            data = json.loads(path.read_text())

        models = [ModelCfg(**m) for m in data["models"]]
        return cls(
            models=models,
            judge_model=data.get("judge_model", "gpt-3.5-turbo"),
            thresholds=data.get("thresholds", {}),
        )