# -----------------------------------------------------------------------------
# src/emafusion/datasets.py
# -----------------------------------------------------------------------------
"""Utility loaders for evaluation datasets (GSM8K, ARC, etc.)."""
from __future__ import annotations

import json
import pathlib
from typing import List, Dict

_DATA_DIR = pathlib.Path("datasets")


def load_dataset(name: str) -> List[Dict]:
    """Return list of {prompt, answer} dicts."""
    path = _DATA_DIR / f"{name}.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    return [json.loads(l) for l in path.read_text().splitlines()]