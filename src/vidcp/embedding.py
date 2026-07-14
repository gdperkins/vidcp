"""Shared sentence-transformers loader (cached across embed + search)."""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=2)
def load_model(name: str):
    """Load (and cache) a SentenceTransformer so embed and search share it.

    Cache size 2: hybrid search and ingest both use the text (MiniLM) model
    and the CLIP model together, so both stay resident instead of evicting
    each other on every alternating call.
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name)
