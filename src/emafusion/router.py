# -----------------------------------------------------------------------------
# src/emafusion/router.py
# -----------------------------------------------------------------------------
"""Routing modules – Taxonomy & Learned routers (Sec. 3 & 4 of paper)."""
from __future__ import annotations

import json
import pathlib
from functools import lru_cache
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
import joblib

_EMBED_MODEL_NAME = "all-MiniLM-L6-v2"


class TaxonomyRouter:
    """Fast exact‑nearest‑neighbour router using pre‑labeled taxonomy vectors."""

    def __init__(self, index_path: str | pathlib.Path, k: int = 3):
        self.index_path = pathlib.Path(index_path)
        if not self.index_path.exists():
            raise FileNotFoundError(self.index_path)

        packed = np.load(self.index_path, allow_pickle=True)
        self.labels: List[str] = packed["labels"].tolist()
        self.vectors: np.ndarray = packed["vectors"]
        self.nn = NearestNeighbors(n_neighbors=k, metric="cosine").fit(self.vectors)
        self.embedder = self._load_embedder()

    @staticmethod
    @lru_cache(maxsize=1)
    def _load_embedder():
        return SentenceTransformer(_EMBED_MODEL_NAME)

    def route(self, prompt: str) -> str:
        vec = self.embedder.encode([prompt], normalize_embeddings=True)
        dists, idx = self.nn.kneighbors(vec, return_distance=True)
        # Majority vote among k nearest labels
        nearest = [self.labels[i] for i in idx[0]]
        return max(set(nearest), key=nearest.count)


class LearnedRouter:
    """Small classifier trained on ambiguous prompts (see paper §4)."""

    def __init__(self, ckpt: str | pathlib.Path):
        ckpt_path = pathlib.Path(ckpt)
        if not ckpt_path.exists():
            raise FileNotFoundError(ckpt)
        bundle = joblib.load(ckpt_path)
        self.clf: LogisticRegression = bundle["model"]
        self.vectorizer: SentenceTransformer = bundle["embedder"]

    def predict(self, prompt: str) -> str:
        vec = self.vectorizer.encode([prompt], normalize_embeddings=True)
        return self.clf.predict(vec)[0]