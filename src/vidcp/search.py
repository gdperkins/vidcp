"""Hybrid search: FTS keyword, text vector, and CLIP visual legs fused with RRF.

The three legs are combined with Reciprocal Rank Fusion so keyword-exact,
semantically-similar, and visually-similar hits all surface, ranked together.
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
    frame_path: str | None = None


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


def _has_frame_vectors(conn, video_id) -> bool:
    sql = "SELECT 1 FROM vec_frames"
    params: list = []
    if video_id:
        sql += " WHERE video_id = ?"
        params.append(video_id)
    return conn.execute(sql + " LIMIT 1", params).fetchone() is not None


def _vec_leg(conn, query, video_id, kind, embed_model):
    # A term-less query (empty / punctuation-only) matches nothing — don't pay
    # the model-load cost or return arbitrary nearest neighbours.
    if not _tokens(query):
        return []
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


def _visual_leg(conn, query, video_id, kind, clip_model):
    # Only contributes when visual hits are wanted and exist; loading the CLIP
    # model is deferred past every cheap bail-out. Anything other than the
    # unfiltered case or an explicit "visual" request is excluded, so an
    # unvalidated caller passing a bogus kind doesn't get visual-only results.
    if kind not in (None, "visual"):
        return []
    if not _tokens(query):
        return []
    if not _has_frame_vectors(conn, video_id):
        return []
    import sqlite_vec

    qvec = load_model(clip_model).encode([query], normalize_embeddings=True)[0]
    sql = (
        f"SELECT frame_id AS ref_id, 'visual' AS kind, video_id, ts_s, distance "
        f"FROM vec_frames WHERE embedding MATCH ? AND k = {_LEG_LIMIT}"
    )
    params: list = [sqlite_vec.serialize_float32(qvec)]
    if video_id:
        sql += " AND video_id = ?"
        params.append(video_id)
    return conn.execute(sql, params).fetchall()


def _rrf(*legs):
    scores: dict[tuple, float] = {}
    for leg in legs:
        for rank, row in enumerate(leg):
            key = (row["video_id"], row["kind"], row["ref_id"])
            scores[key] = scores.get(key, 0.0) + 1.0 / (_RRF_K + rank + 1)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def _snippet(text: str, query: str) -> str:
    # Locate against the original text (case-insensitively) so the slice indices
    # stay valid even when lower-casing changes length (e.g. Turkish İ, ligatures).
    pos = -1
    for token in _tokens(query):
        match = re.search(re.escape(token), text, re.IGNORECASE)
        if match:
            pos = match.start()
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
    if kind == "visual":
        return conn.execute(
            "SELECT '' AS text, ts_s AS start_s, path FROM frames WHERE id = ?", (ref_id,)
        ).fetchone()
    table = "segments" if kind == "transcript" else "ocr_blocks"
    return conn.execute(f"SELECT text, start_s FROM {table} WHERE id = ?", (ref_id,)).fetchone()


def search(
    conn: sqlite3.Connection,
    query: str,
    video_id: str | None = None,
    kind: str | None = None,
    limit: int = 10,
    embed_model: str | None = None,
    clip_model: str | None = None,
) -> list[Hit]:
    if embed_model is None or clip_model is None:
        from vidcp.config import get_settings

        settings = get_settings()
        embed_model = embed_model or settings.embed_model
        clip_model = clip_model or settings.clip_model

    ranked = _rrf(
        _fts_leg(conn, query, video_id, kind),
        _vec_leg(conn, query, video_id, kind, embed_model),
        _visual_leg(conn, query, video_id, kind, clip_model),
    )

    hits: list[Hit] = []
    for (vid, hit_kind, ref_id), score in ranked:
        if len(hits) >= limit:
            break
        row = _hydrate(conn, (vid, hit_kind, ref_id))
        if row is None:
            continue
        if hit_kind == "visual":
            snippet = f"visual match at {row['start_s']:.1f}s"
            frame_path = row["path"]
        else:
            snippet = _snippet(row["text"], query)
            frame_path = None
        hits.append(
            Hit(
                video_id=vid,
                short_id=vid[:8],
                kind=hit_kind,
                ref_id=ref_id,
                ts_s=row["start_s"],
                text=row["text"],
                snippet=snippet,
                score=score,
                frame_path=frame_path,
            )
        )
    return hits
