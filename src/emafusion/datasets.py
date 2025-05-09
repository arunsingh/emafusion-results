# -----------------------------------------------------------------------------
# src/emafusion/datasets.py
# -----------------------------------------------------------------------------
"""Utility loaders for evaluation datasets (GSM8K, ARC, etc.)."""
"""Helper for loading JSONL prompt files."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def load_jsonl(path: Path) -> List[Dict[str, Any]]:  # noqa: D401 – simple docstring
    if not path.exists():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text().splitlines()]
