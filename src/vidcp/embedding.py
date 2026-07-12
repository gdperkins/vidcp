"""Shared sentence-transformers loader (cached across embed + search)."""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def load_model(name: str):
    """Load (and cache) a SentenceTransformer so embed and search share it."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name)
