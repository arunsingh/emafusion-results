# -----------------------------------------------------------------------------
# src/emafusion/fusion.py
# -----------------------------------------------------------------------------
"""Answer fusion – pick best among multiple drafts (paper§6)."""
"""Optional fusion stage if multiple drafts survive judging."""
from __future__ import annotations

from typing import List, Tuple


def majority_vote(candidates: List[Tuple[str, float]]) -> str:
    """Return answer with highest *confidence*; tie‑break shorter answer."""

    if not candidates:
        raise ValueError("No candidates to fuse.")
    # Sort by (‑confidence, len(answer)) so highest confidence, then shortest wins.
    candidates = sorted(candidates, key=lambda x: (-x[1], len(x[0])))
    return candidates[0][0]
    