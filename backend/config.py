"""Loads api.env and exposes typed config."""
from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / "api.env"

load_dotenv(ENV_PATH)


def _get(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


@dataclass(frozen=True)
class Settings:
    sarvam_api_key: str
    sarvam_model: str
    sarvam_base_url: str

    gemini_api_key: str
    gemini_vision_model: str
    gemini_embed_model: str
    gemini_rewriter_model: str

    cohere_api_key: str
    cohere_rerank_model: str

    whapi_token: str
    whapi_base_url: str

    host: str
    port: int
    public_base_url: str

    manuals_dir: Path
    chroma_dir: Path
    bm25_index_path: Path


def load_settings() -> Settings:
    return Settings(
        sarvam_api_key=_get("SARVAM_API_KEY"),
        sarvam_model=_get("SARVAM_MODEL", "sarvam-m"),
        sarvam_base_url=_get("SARVAM_BASE_URL", "https://api.sarvam.ai"),
        gemini_api_key=_get("GEMINI_API_KEY"),
        gemini_vision_model=_get("GEMINI_VISION_MODEL", "gemini-2.0-flash-exp"),
        gemini_embed_model=_get("GEMINI_EMBED_MODEL", "gemini-embedding-001"),
        gemini_rewriter_model=_get("GEMINI_REWRITER_MODEL", "gemini-2.0-flash-exp"),
        cohere_api_key=_get("COHERE_API_KEY"),
        cohere_rerank_model=_get("COHERE_RERANK_MODEL", "rerank-v3.5"),
        whapi_token=_get("WHAPI_TOKEN"),
        whapi_base_url=_get("WHAPI_BASE_URL", "https://gate.whapi.cloud"),
        host=_get("HOST", "0.0.0.0"),
        port=int(_get("PORT", "8000")),
        public_base_url=_get("PUBLIC_BASE_URL", "http://localhost:8000"),
        manuals_dir=PROJECT_ROOT / _get("MANUALS_DIR", "data/manuals"),
        chroma_dir=PROJECT_ROOT / _get("CHROMA_DIR", "data/chroma_db"),
        bm25_index_path=PROJECT_ROOT / _get("BM25_INDEX_PATH", "data/bm25_index.pkl"),
    )


settings = load_settings()
