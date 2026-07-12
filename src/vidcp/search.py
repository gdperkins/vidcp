"""Hybrid search: an FTS keyword leg and a vector leg fused with RRF.

The two legs are combined with Reciprocal Rank Fusion so keyword-exact and
semantically-similar hits both surface, ranked together.
"""

from __future__ import annotations

import re
import sqlite3

from pydantic import BaseModel

from vidcp.embedding import load_model

_RRF_K = 60
_LEG_LIMIT = 50
_SNIPPET_RADIUS = 80
_SNIPPET_FALLBACK = 160


class Hit(BaseModel):
    video_id: str
    short_id: str
    kind: str
    ref_id: int
    ts_s: float
    text: str
    snippet: str
    score: float


def _tokens(query: str) -> list[str]:
    return re.findall(r"\w+", query.lower())


def _fts_query(query: str) -> str:
    # Quote each token so FTS5 never chokes on ':' / '-' etc.
    return " ".join(f'"{token}"' for token in _tokens(query))


def _fts_leg(conn, query, video_id, kind):
    match = _fts_query(query)
    if not match:
        return []
    sql = "SELECT ref_id, kind, video_id, ts_s, bm25(fts) AS s FROM fts WHERE fts MATCH ?"
    params: list = [match]
    if video_id:
        sql += " AND video_id = ?"
        params.append(video_id)
    if kind:
        sql += " AND kind = ?"
        params.append(kind)
    sql += " ORDER BY s LIMIT ?"
    params.append(_LEG_LIMIT)
    return conn.execute(sql, params).fetchall()


def _has_vectors(conn, video_id, kind) -> bool:
    clauses, params = [], []
    if video_id:
        clauses.append("video_id = ?")
        params.append(video_id)
    if kind:
        clauses.append("kind = ?")
        params.append(kind)
    sql = "SELECT 1 FROM vec"
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " LIMIT 1"
    return conn.execute(sql, params).fetchone() is not None


def _vec_leg(conn, query, video_id, kind, embed_model):
    # Skip the (model-loading) vector leg when there is nothing to match.
    if not _has_vectors(conn, video_id, kind):
        return []
    import sqlite_vec

    qvec = load_model(embed_model).encode([query], normalize_embeddings=True)[0]
    sql = (
        f"SELECT ref_id, kind, video_id, ts_s, distance FROM vec "
        f"WHERE embedding MATCH ? AND k = {_LEG_LIMIT}"
    )
    params: list = [sqlite_vec.serialize_float32(qvec)]
    if video_id:
        sql += " AND video_id = ?"
        params.append(video_id)
    if kind:
        sql += " AND kind = ?"
        params.append(kind)
    return conn.execute(sql, params).fetchall()


def _rrf(fts_rows, vec_rows):
    scores: dict[tuple, float] = {}
    for leg in (fts_rows, vec_rows):
        for rank, row in enumerate(leg):
            key = (row["video_id"], row["kind"], row["ref_id"])
            scores[key] = scores.get(key, 0.0) + 1.0 / (_RRF_K + rank + 1)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def _snippet(text: str, query: str) -> str:
    low = text.lower()
    pos = -1
    for token in _tokens(query):
        found = low.find(token)
        if found != -1:
            pos = found
            break
    if pos == -1:
        return text[:_SNIPPET_FALLBACK].strip()
    start = max(0, pos - _SNIPPET_RADIUS)
    end = min(len(text), pos + _SNIPPET_RADIUS)
    snippet = text[start:end].strip()
    if start > 0:
        snippet = "…" + snippet
    if end < len(text):
        snippet = snippet + "…"
    return snippet


def _hydrate(conn, key):
    video_id, kind, ref_id = key
    table = "segments" if kind == "transcript" else "ocr_blocks"
    row = conn.execute(f"SELECT text, start_s FROM {table} WHERE id = ?", (ref_id,)).fetchone()
    return row


def search(
    conn: sqlite3.Connection,
    query: str,
    video_id: str | None = None,
    kind: str | None = None,
    limit: int = 10,
    embed_model: str | None = None,
) -> list[Hit]:
    if embed_model is None:
        from vidcp.config import get_settings

        embed_model = get_settings().embed_model

    ranked = _rrf(
        _fts_leg(conn, query, video_id, kind),
        _vec_leg(conn, query, video_id, kind, embed_model),
    )

    hits: list[Hit] = []
    for (vid, hit_kind, ref_id), score in ranked:
        if len(hits) >= limit:
            break
        row = _hydrate(conn, (vid, hit_kind, ref_id))
        if row is None:
            continue
        hits.append(
            Hit(
                video_id=vid,
                short_id=vid[:8],
                kind=hit_kind,
                ref_id=ref_id,
                ts_s=row["start_s"],
                text=row["text"],
                snippet=_snippet(row["text"], query),
                score=score,
            )
        )
    return hits
