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
from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import List, Dict, Any

import yaml


@dataclass
class ModelCfg:
    name: str
    provider: str  # "openai" | "ollama" | "vertexai" …
    price_prompt: float  # USD per 1k tokens
    price_completion: float  # USD per 1k tokens
    max_tokens: int = 4096


@dataclass
class Settings:
    models: List[ModelCfg]
    taxonomy_index: pathlib.Path
    learned_ckpt: pathlib.Path
    judge_model: str
    thresholds: Dict[str, float]


def load_config(path: str | pathlib.Path) -> Settings:
    data: Dict[str, Any] = yaml.safe_load(pathlib.Path(path).read_text())
    models = [ModelCfg(**m) for m in data["models"]]
    return Settings(
        models=models,
        taxonomy_index=pathlib.Path(data["taxonomy_index"]),
        learned_ckpt=pathlib.Path(data["learned_ckpt"]),
        judge_model=data["judge_model"],
        thresholds=data.get("thresholds", {"confidence": 0.6}),
    )