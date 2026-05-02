"""PDF ingestion pipeline.

Steps:
  1. Parse PDF page-by-page (text + tables).
  2. Group lines into section-aware chunks (~400-600 tokens).
  3. Tag with component/symptom keywords.
  4. Embed with Gemini gemini-embedding-001.
  5. Store in ChromaDB + BM25 keyword index.
"""
from __future__ import annotations

import hashlib
import pickle
import re
from pathlib import Path
from typing import Iterable

import chromadb
from rank_bm25 import BM25Okapi

from .config import settings
from .models import Chunk

# ---------------------------------------------------------------------------
# Domain vocabulary — used for metadata enrichment.
# Extend these lists as you ingest more manuals.
# ---------------------------------------------------------------------------
COMPONENT_KEYWORDS = {
    "engine": ["engine", "piston", "cylinder", "crankshaft", "camshaft", "valve"],
    "cooling": ["coolant", "radiator", "fan", "thermostat", "cooling"],
    "fuel": ["fuel", "carburettor", "carburetor", "injector", "petrol", "tank"],
    "exhaust": ["exhaust", "muffler", "silencer", "emission", "smoke"],
    "electrical": ["battery", "alternator", "spark plug", "ignition", "wiring", "fuse"],
    "transmission": ["clutch", "gear", "transmission", "gearbox", "chain", "sprocket"],
    "brakes": ["brake", "disc", "pad", "caliper", "abs"],
    "suspension": ["suspension", "fork", "shock", "spring"],
    "tyres": ["tyre", "tire", "wheel", "tube", "puncture"],
    "lubrication": ["oil", "lubricant", "grease", "lube"],
}

SYMPTOM_KEYWORDS = {
    "smoke": ["smoke", "smoking", "fume"],
    "noise": ["noise", "knock", "rattle", "click", "tick", "whine", "squeal"],
    "leak": ["leak", "leaking", "drip", "seep"],
    "vibration": ["vibration", "shake", "wobble"],
    "starting": ["start", "starting", "stall", "stalling", "crank", "no start"],
    "overheating": ["overheat", "overheating", "hot"],
    "misfire": ["misfire", "missing", "sputter"],
}

# Headings that suggest a new logical section in service manuals.
SECTION_HEADING_RE = re.compile(
    r"^(?:\d+(?:\.\d+)*\s+)?[A-Z][A-Z0-9 \-/&]{3,}$"
)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def _extract_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """Return [(page_number, text), ...]. Tables are flattened into the text."""
    import pdfplumber  # lazy — only needed when actually parsing a PDF
    pages: list[tuple[int, str]] = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            parts: list[str] = []
            text = page.extract_text() or ""
            if text:
                parts.append(text)
            for table in page.extract_tables() or []:
                for row in table:
                    cleaned = [c.strip() if c else "" for c in row]
                    parts.append(" | ".join(cleaned))
            pages.append((i, "\n".join(parts)))
    return pages


# ---------------------------------------------------------------------------
# Chunking — section-aware + token-bounded
# ---------------------------------------------------------------------------
def _approx_tokens(s: str) -> int:
    return max(1, len(s) // 4)


def _chunk_page(page_num: int, text: str, current_section: str,
                max_tokens: int = 500) -> tuple[list[tuple[int, str, str]], str]:
    """Yield (page, section, chunk_text) tuples for one page.

    Returns the (chunks, updated_section) tuple — the section name
    persists across pages until a new heading is found.
    """
    chunks: list[tuple[int, str, str]] = []
    buffer: list[str] = []
    buffer_tokens = 0
    section = current_section

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        # Promote to new section if line looks like a heading.
        if SECTION_HEADING_RE.match(stripped) and len(stripped.split()) <= 8:
            if buffer:
                chunks.append((page_num, section, "\n".join(buffer)))
                buffer, buffer_tokens = [], 0
            section = stripped.title()
            continue

        line_tokens = _approx_tokens(stripped)
        if buffer_tokens + line_tokens > max_tokens and buffer:
            chunks.append((page_num, section, "\n".join(buffer)))
            buffer, buffer_tokens = [], 0
        buffer.append(stripped)
        buffer_tokens += line_tokens

    if buffer:
        chunks.append((page_num, section, "\n".join(buffer)))
    return chunks, section


# ---------------------------------------------------------------------------
# Metadata enrichment
# ---------------------------------------------------------------------------
def _tag(text: str, vocab: dict[str, list[str]]) -> list[str]:
    lowered = text.lower()
    return sorted(
        tag for tag, words in vocab.items()
        if any(w in lowered for w in words)
    )


def _build_chunks(manual_id: str, pages: list[tuple[int, str]]) -> list[Chunk]:
    chunks: list[Chunk] = []
    section = "General"
    seq = 0
    for page_num, text in pages:
        page_chunks, section = _chunk_page(page_num, text, section)
        for page, sect, body in page_chunks:
            if len(body.strip()) < 50:
                continue
            cid = hashlib.sha1(
                f"{manual_id}:{page}:{seq}:{body}".encode()
            ).hexdigest()[:16]
            seq += 1
            chunks.append(Chunk(
                chunk_id=cid,
                manual_id=manual_id,
                page=page,
                section=sect,
                text=body,
                component_tags=_tag(body, COMPONENT_KEYWORDS),
                symptom_tags=_tag(body, SYMPTOM_KEYWORDS),
            ))
    return chunks


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------
def _embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed via Gemini. Falls back to deterministic stub if no key set."""
    if not settings.gemini_api_key:
        # Deterministic pseudo-embedding so the system runs without keys.
        # Replace with real embeddings before any quality testing.
        return [_stub_embedding(t) for t in texts]
    import google.generativeai as genai
    genai.configure(api_key=settings.gemini_api_key)
    out: list[list[float]] = []
    for t in texts:
        resp = genai.embed_content(
            model=f"models/{settings.gemini_embed_model}",
            content=t,
            task_type="retrieval_document",
        )
        out.append(resp["embedding"])
    return out


def _stub_embedding(text: str, dim: int = 768) -> list[float]:
    """Hash-based pseudo-embedding for offline/no-key dev only."""
    import math
    h = hashlib.sha256(text.encode()).digest()
    vals = [b / 255.0 - 0.5 for b in (h * ((dim // len(h)) + 1))[:dim]]
    norm = math.sqrt(sum(v * v for v in vals)) or 1.0
    return [v / norm for v in vals]


def _tokenise(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _dedupe_by_id(chunks: list[Chunk]) -> list[Chunk]:
    """Keep first occurrence of each chunk_id. Last line of defence against collisions."""
    seen: set[str] = set()
    out: list[Chunk] = []
    for c in chunks:
        if c.chunk_id in seen:
            continue
        seen.add(c.chunk_id)
        out.append(c)
    return out


def _save_bm25(chunks: list[Chunk]) -> None:
    chunks = _dedupe_by_id(chunks)
    docs = [_tokenise(c.text) for c in chunks]
    bm25 = BM25Okapi(docs)
    settings.bm25_index_path.parent.mkdir(parents=True, exist_ok=True)
    with open(settings.bm25_index_path, "wb") as f:
        pickle.dump({
            "bm25": bm25,
            "chunk_ids": [c.chunk_id for c in chunks],
            "chunks": [c.model_dump() for c in chunks],
        }, f)


def _save_chroma(chunks: list[Chunk]) -> None:
    chunks = _dedupe_by_id(chunks)
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(settings.chroma_dir),
        settings=chromadb.config.Settings(anonymized_telemetry=False),
    )
    collection = client.get_or_create_collection("manuals")
    embeddings = _embed_batch([c.text for c in chunks])
    collection.upsert(
        ids=[c.chunk_id for c in chunks],
        embeddings=embeddings,
        documents=[c.text for c in chunks],
        metadatas=[{
            "manual_id": c.manual_id,
            "page": c.page,
            "section": c.section,
            "components": ",".join(c.component_tags),
            "symptoms": ",".join(c.symptom_tags),
        } for c in chunks],
    )


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------
def ingest_pdf(pdf_path: Path, manual_id: str) -> dict:
    """Ingest one manual PDF end-to-end."""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    pages = _extract_pages(pdf_path)
    chunks = _build_chunks(manual_id, pages)
    if not chunks:
        return {"manual_id": manual_id, "pages": len(pages), "chunks": 0}

    _save_chroma(chunks)
    _save_bm25(_load_all_chunks_plus(chunks))
    return {"manual_id": manual_id, "pages": len(pages), "chunks": len(chunks)}


def _load_all_chunks_plus(new_chunks: list[Chunk]) -> list[Chunk]:
    """Merge new chunks with existing BM25 corpus so we keep all manuals indexed."""
    existing: list[Chunk] = []
    if settings.bm25_index_path.exists():
        with open(settings.bm25_index_path, "rb") as f:
            data = pickle.load(f)
        existing = [Chunk(**c) for c in data.get("chunks", [])]
    new_ids = {c.chunk_id for c in new_chunks}
    return [c for c in existing if c.chunk_id not in new_ids] + new_chunks
