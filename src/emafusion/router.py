# -----------------------------------------------------------------------------
# src/emafusion/router.py
# -----------------------------------------------------------------------------
"""Routing modules – Taxonomy & Learned routers (Sec. 3 & 4 of paper)."""
"""Routers pick a *candidate model* for a prompt before the cascade starts."""
from __future__ import annotations

import pathlib
from functools import lru_cache
from typing import List

import numpy as np  # type: ignore
from sentence_transformers import SentenceTransformer, util  # type: ignore


class TaxonomyRouter:
    """K‑NN lookup on a curated embedding index."""

    def __init__(self, index_path: pathlib.Path):
        self._index_path = index_path
        self._load()

    def _load(self):
        import pickle

        with open(self._index_path, "rb") as fh:
            obj = pickle.load(fh)
        self._emb = obj["embeddings"]  # shape: (N, d)
        self._tasks = obj["tasks"]  # list[{task, model}]
        self._encoder = SentenceTransformer(obj["encoder_name"])

    @lru_cache(maxsize=128)
    def route(self, prompt: str) -> str | None:  # noqa: D401
        vec = self._encoder.encode(prompt, normalize_embeddings=True)
        sims = util.dot_score(vec, self._emb)[0]
        idx = int(np.argmax(sims))
        if float(sims[idx]) > 0.80:
            return self._tasks[idx]["model"]
        return None


class LearnedRouter:
    """Text‑classifier router (scikit‑learn / HF) trained on task→model labels."""

    def __init__(self, ckpt: pathlib.Path):
        import joblib  # type: ignore

        self._clf = joblib.load(ckpt)

    def predict(self, prompt: str) -> str:  # noqa: D401
        return self._clf.predict([prompt])[0]
