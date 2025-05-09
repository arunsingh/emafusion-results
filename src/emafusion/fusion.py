# -----------------------------------------------------------------------------
# src/emafusion/fusion.py
# -----------------------------------------------------------------------------
"""Answer fusion – pick best among multiple drafts (paper §6)."""
from __future__ import annotations

from collections import Counter
from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer, util

_embed = SentenceTransformer("all-MiniLM-L6-v2")


def fuse_answers(prompt: str, drafts: List[Tuple[str, float]]) -> str:
    """Given list of (draft, score) choose final answer.

    Strategy: (1) highest judge score; (2) tie‑break with semantic consensus.
    """
    drafts_sorted = sorted(drafts, key=lambda x: x[1], reverse=True)
    top_score = drafts_sorted[0][1]
    winners = [d for d, s in drafts_sorted if s == top_score]
    if len(winners) == 1:
        return winners[0]

    # Tie – choose answer most similar to others (centroid)
    embs = _embed.encode(winners, normalize_embeddings=True)
    sims = util.cos_sim(embs, embs).sum(axis=1).A1
    return winners[int(np.argmax(sims))]
