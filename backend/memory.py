"""Per-session conversation memory.

Stores recent (user, assistant) turn pairs in SQLite, keyed by session_id.
session_id is:
  - the WhatsApp sender phone for WhatsApp users
  - a browser-generated UUID for web users (sent from the frontend)
  - "anonymous" if the client doesn't supply one

Memory is included in the Sarvam prompt as previous chat messages so that
follow-up questions like "and how do I fix it?" resolve correctly.
"""
from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from .config import settings


_DB_PATH = settings.chroma_dir.parent / "memory.db"
_MAX_TURNS = 5  # Keep last 5 user+assistant pairs (10 messages total).


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
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                ts INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                language TEXT
            )
        """)
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_session_ts ON conversations(session_id, ts)"
        )


_init()


def _normalize_history(rows: list[sqlite3.Row]) -> list[dict]:
    """Keep only well-formed alternating user/assistant turns starting with user."""
    normalized: list[dict] = []
    expected_role = "user"
    for row in rows:
        role = row["role"]
        content = row["content"]
        if role != expected_role or not content:
            continue
        normalized.append({"role": role, "content": content})
        expected_role = "assistant" if expected_role == "user" else "user"

    if normalized and normalized[-1]["role"] != "assistant":
        normalized.pop()
    return normalized


def get_history(session_id: Optional[str], max_turns: int = _MAX_TURNS) -> list[dict]:
    """Return the last `max_turns` user+assistant turn pairs as Sarvam-shaped messages."""
    if not session_id:
        return []
    with _conn() as c:
        rows = c.execute(
            "SELECT role, content FROM conversations "
            "WHERE session_id = ? ORDER BY ts DESC LIMIT ?",
            (session_id, max_turns * 2),
        ).fetchall()
    rows = list(reversed(rows))
    return _normalize_history(rows)


def append_turn(session_id: Optional[str], role: str, content: str,
                language: Optional[str] = None) -> None:
    if not session_id or not content:
        return
    with _conn() as c:
        c.execute(
            "INSERT INTO conversations (session_id, ts, role, content, language) "
            "VALUES (?, ?, ?, ?, ?)",
            (session_id, int(time.time() * 1000), role, content, language),
        )


def clear_session(session_id: str) -> int:
    with _conn() as c:
        cur = c.execute("DELETE FROM conversations WHERE session_id = ?", (session_id,))
        return cur.rowcount
