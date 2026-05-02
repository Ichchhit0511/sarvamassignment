"""Per-query metrics — captures latency, token usage, and accuracy proxies.

We record one row per /api/query invocation in SQLite. The dashboard queries
this table for live charts and aggregates.

What "accuracy" means here (no single number; we track several proxies):
  - manual_supported_rate  : % of queries Sarvam answered confidently from manual
  - citations_kept_rate    : % of Sarvam citations that survived the verifier
                             (cited chunk_id/page actually appeared in retrieval)
  - top_retrieval_score    : Cohere rerank score of the top-1 chunk (precision proxy)
  - confidence_distribution: high/medium/low spread

For ground-truth accuracy you run scripts/evaluate.py against a golden test set.
"""
from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from typing import Optional

from .config import settings


_DB_PATH = settings.chroma_dir.parent / "metrics.db"


@contextmanager
def _conn():
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(_DB_PATH)
    c.row_factory = sqlite3.Row
    try:
        yield c
        c.commit()
    finally:
        c.close()


def _init() -> None:
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS queries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts INTEGER NOT NULL,
                session_id TEXT,
                manual_id TEXT,
                query TEXT,
                language TEXT,

                has_image INTEGER,
                num_rewrites INTEGER,
                num_retrieved INTEGER,
                top_retrieval_score REAL,

                manual_supported INTEGER,
                confidence TEXT,
                num_citations_raw INTEGER,
                num_citations_kept INTEGER,

                sarvam_input_tokens INTEGER,
                sarvam_output_tokens INTEGER,
                gemini_input_tokens INTEGER,
                gemini_output_tokens INTEGER,

                vision_ms INTEGER,
                rewrite_ms INTEGER,
                retrieve_ms INTEGER,
                generate_ms INTEGER,
                verify_ms INTEGER,
                total_ms INTEGER,

                error TEXT
            )
        """)


_init()


def record(**fields) -> None:
    fields["ts"] = int(time.time() * 1000)
    keys = ", ".join(fields.keys())
    placeholders = ", ".join("?" * len(fields))
    values = list(fields.values())
    with _conn() as c:
        c.execute(f"INSERT INTO queries ({keys}) VALUES ({placeholders})", values)


def recent(limit: int = 50) -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM queries ORDER BY ts DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def aggregate(window_hours: int = 24) -> dict:
    """Return summary stats over the last N hours."""
    cutoff = int((time.time() - window_hours * 3600) * 1000)
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM queries WHERE ts >= ? AND error IS NULL", (cutoff,)
        ).fetchall()

    if not rows:
        return {
            "total_queries": 0,
            "manual_supported_rate": 0.0,
            "citations_kept_rate": 0.0,
            "avg_top_score": 0.0,
            "avg_total_ms": 0,
            "avg_sarvam_input_tokens": 0,
            "avg_sarvam_output_tokens": 0,
            "avg_gemini_input_tokens": 0,
            "avg_gemini_output_tokens": 0,
            "stage_latency_ms": {},
            "confidence": {"high": 0, "medium": 0, "low": 0},
            "languages": {},
        }

    n = len(rows)
    supported = sum(1 for r in rows if r["manual_supported"])
    raw_cites = sum(r["num_citations_raw"] or 0 for r in rows)
    kept_cites = sum(r["num_citations_kept"] or 0 for r in rows)

    confidence: dict[str, int] = {"high": 0, "medium": 0, "low": 0}
    languages: dict[str, int] = {}
    for r in rows:
        c_key = (r["confidence"] or "low").lower()
        confidence[c_key] = confidence.get(c_key, 0) + 1
        lang = r["language"] or "unknown"
        languages[lang] = languages.get(lang, 0) + 1

    def avg(field: str) -> float:
        vals = [r[field] for r in rows if r[field] is not None]
        return sum(vals) / len(vals) if vals else 0

    return {
        "total_queries": n,
        "manual_supported_rate": supported / n,
        "citations_kept_rate": (kept_cites / raw_cites) if raw_cites else 1.0,
        "avg_top_score": round(avg("top_retrieval_score"), 4),
        "avg_total_ms": int(avg("total_ms")),
        "avg_sarvam_input_tokens": int(avg("sarvam_input_tokens")),
        "avg_sarvam_output_tokens": int(avg("sarvam_output_tokens")),
        "avg_gemini_input_tokens": int(avg("gemini_input_tokens")),
        "avg_gemini_output_tokens": int(avg("gemini_output_tokens")),
        "stage_latency_ms": {
            "vision": int(avg("vision_ms")),
            "rewrite": int(avg("rewrite_ms")),
            "retrieve": int(avg("retrieve_ms")),
            "generate": int(avg("generate_ms")),
            "verify": int(avg("verify_ms")),
        },
        "confidence": confidence,
        "languages": languages,
    }
