"""Hybrid retrieval: vector (Chroma) + BM25 + RRF fusion + optional Cohere rerank."""
from __future__ import annotations

import pickle
import re
from typing import Optional

import chromadb

from .config import settings
from .models import Chunk, RetrievedChunk


def _tokenise(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


# ---------------------------------------------------------------------------
# BM25 search
# ---------------------------------------------------------------------------
def _load_bm25() -> Optional[dict]:
    if not settings.bm25_index_path.exists():
        return None
    with open(settings.bm25_index_path, "rb") as f:
        return pickle.load(f)


def bm25_search(queries: list[str], top_k: int = 10,
                manual_id: Optional[str] = None) -> list[tuple[str, float]]:
    data = _load_bm25()
    if not data:
        return []
    bm25 = data["bm25"]
    chunk_ids: list[str] = data["chunk_ids"]
    chunks: list[Chunk] = [Chunk(**c) for c in data["chunks"]]

    aggregated: dict[str, float] = {}
    for q in queries:
        scores = bm25.get_scores(_tokenise(q))
        for cid, score, ch in zip(chunk_ids, scores, chunks):
            if manual_id and ch.manual_id != manual_id:
                continue
            if score <= 0:
                continue
            aggregated[cid] = max(aggregated.get(cid, 0.0), float(score))

    ranked = sorted(aggregated.items(), key=lambda kv: kv[1], reverse=True)
    return ranked[:top_k]


# ---------------------------------------------------------------------------
# Vector search
# ---------------------------------------------------------------------------
def _embed_query(text: str) -> list[float]:
    if not settings.gemini_api_key:
        from .ingest import _stub_embedding
        return _stub_embedding(text)
    import google.generativeai as genai
    genai.configure(api_key=settings.gemini_api_key)
    resp = genai.embed_content(
        model=f"models/{settings.gemini_embed_model}",
        content=text,
        task_type="retrieval_query",
    )
    return resp["embedding"]


def vector_search(queries: list[str], top_k: int = 10,
                  manual_id: Optional[str] = None) -> list[tuple[str, float]]:
    if not settings.chroma_dir.exists():
        return []
    client = chromadb.PersistentClient(
        path=str(settings.chroma_dir),
        settings=chromadb.config.Settings(anonymized_telemetry=False),
    )
    try:
        collection = client.get_collection("manuals")
    except Exception:
        return []

    where = {"manual_id": manual_id} if manual_id else None
    aggregated: dict[str, float] = {}
    for q in queries:
        emb = _embed_query(q)
        res = collection.query(
            query_embeddings=[emb],
            n_results=top_k,
            where=where,
        )
        ids = res.get("ids", [[]])[0]
        dists = res.get("distances", [[]])[0]
        for cid, dist in zip(ids, dists):
            # Lower cosine distance is better — flip to similarity.
            sim = 1.0 - float(dist)
            aggregated[cid] = max(aggregated.get(cid, 0.0), sim)
    return sorted(aggregated.items(), key=lambda kv: kv[1], reverse=True)[:top_k]


# ---------------------------------------------------------------------------
# Reciprocal rank fusion
# ---------------------------------------------------------------------------
def reciprocal_rank_fusion(*ranked_lists: list[tuple[str, float]],
                           k: int = 60) -> list[tuple[str, float]]:
    fused: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, (cid, _) in enumerate(ranked):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(fused.items(), key=lambda kv: kv[1], reverse=True)


# ---------------------------------------------------------------------------
# Lookup chunk objects from BM25 store (it has the full content)
# ---------------------------------------------------------------------------
def _chunk_lookup() -> dict[str, Chunk]:
    data = _load_bm25()
    if not data:
        return {}
    return {c["chunk_id"]: Chunk(**c) for c in data["chunks"]}


# ---------------------------------------------------------------------------
# Optional Cohere re-ranker
# ---------------------------------------------------------------------------
def _cohere_rerank(query: str, candidates: list[Chunk],
                   top_k: int) -> list[tuple[str, float]]:
    if not settings.cohere_api_key or not candidates:
        return [(c.chunk_id, 1.0 / (i + 1)) for i, c in enumerate(candidates[:top_k])]
    import cohere
    co = cohere.ClientV2(settings.cohere_api_key)
    docs = [c.text for c in candidates]
    resp = co.rerank(
        model=settings.cohere_rerank_model,
        query=query,
        documents=docs,
        top_n=top_k,
    )
    out: list[tuple[str, float]] = []
    for r in resp.results:
        out.append((candidates[r.index].chunk_id, float(r.relevance_score)))
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def retrieve(queries: list[str], original_query: str,
             top_k: int = 5, manual_id: Optional[str] = None,
             pool_size: int = 12) -> list[RetrievedChunk]:
    """End-to-end hybrid retrieval pipeline.

    queries           — list of rewritten search queries (use [original] if no rewriter).
    original_query    — the user's actual question, used for re-ranking.
    top_k             — final number of chunks to return.
    manual_id         — restrict to one manual.
    """
    bm25_hits = bm25_search(queries, top_k=pool_size, manual_id=manual_id)
    vec_hits = vector_search(queries, top_k=pool_size, manual_id=manual_id)
    fused = reciprocal_rank_fusion(bm25_hits, vec_hits)

    lookup = _chunk_lookup()
    candidates = [lookup[cid] for cid, _ in fused if cid in lookup][:pool_size]
    if not candidates:
        return []

    reranked = _cohere_rerank(original_query, candidates, top_k=top_k)
    return [
        RetrievedChunk(chunk=lookup[cid], score=score)
        for cid, score in reranked
        if cid in lookup
    ]
