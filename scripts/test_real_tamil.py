"""REAL end-to-end test using the keys in api.env.

Pipeline (every stage hits real APIs):
  1. Index 4 synthetic English manual chunks → real Gemini embeddings + BM25
  2. Real Gemini rewriter expands the Tamil query into English search phrases
  3. Hybrid retrieval (real multilingual embeddings + BM25 + RRF)
  4. Real Sarvam 105B grounded generation in Tamil
  5. Citation verifier
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import settings
from backend.models import Chunk
from backend.ingest import _save_chroma, _save_bm25
from backend import retrieval
from backend.query_rewriter import rewrite
from backend.generator import generate
from backend.verifier import verify


def banner(t: str) -> None:
    print("\n" + "=" * 70 + f"\n  {t}\n" + "=" * 70)


def assert_keys() -> None:
    missing = []
    if not settings.sarvam_api_key: missing.append("SARVAM_API_KEY")
    if not settings.gemini_api_key: missing.append("GEMINI_API_KEY")
    if missing:
        sys.exit(f"❌ Missing keys in api.env: {missing}")
    print(f"  Sarvam model: {settings.sarvam_model}")
    print(f"  Gemini embed: {settings.gemini_embed_model}")
    print(f"  Gemini rewriter: {settings.gemini_rewriter_model}")


def main() -> None:
    banner("Pre-flight — keys present?")
    assert_keys()

    # Reset state so the test is reproducible.
    if settings.chroma_dir.exists():
        shutil.rmtree(settings.chroma_dir)
    if settings.bm25_index_path.exists():
        settings.bm25_index_path.unlink()

    chunks = [
        Chunk(
            chunk_id="c001", manual_id="re_classic_350", page=42,
            section="Troubleshooting > Engine > White Smoke",
            text=(
                "WHITE SMOKE FROM EXHAUST. Possible causes: "
                "(1) Coolant entering combustion chamber due to head gasket failure. "
                "(2) Condensation in exhaust during cold start (normal, dissipates "
                "in 2-3 minutes). Action: If smoke persists after engine warms up, "
                "stop riding and consult an authorized service center. Do not continue "
                "operation as this may damage the catalytic converter."
            ),
            component_tags=["engine", "cooling", "exhaust"], symptom_tags=["smoke"],
        ),
        Chunk(
            chunk_id="c002", manual_id="re_classic_350", page=43,
            section="Troubleshooting > Engine > Black Smoke",
            text=(
                "BLACK SMOKE FROM EXHAUST. Indicates rich fuel mixture. "
                "Check air filter for clogging. Replace if dirty. "
                "Verify fuel injector cleanliness. Persistent black smoke requires "
                "ECU diagnostic at service center."
            ),
            component_tags=["engine", "fuel", "exhaust"], symptom_tags=["smoke"],
        ),
        Chunk(
            chunk_id="c003", manual_id="re_classic_350", page=18,
            section="Maintenance > Cooling System",
            text=(
                "COOLANT LEVEL CHECK. Inspect coolant reservoir weekly. "
                "Top up with manufacturer-specified coolant only. "
                "Check radiator hoses for cracks or leaks. Capacity: 1.2 litres."
            ),
            component_tags=["cooling"], symptom_tags=["leak"],
        ),
        Chunk(
            chunk_id="c004", manual_id="re_classic_350", page=71,
            section="Brakes > Pad Replacement",
            text=(
                "BRAKE PAD REPLACEMENT. Loosen caliper bolts. Remove old pads. "
                "Inspect rotor for scoring. Install new pads ensuring chamfered edge "
                "faces rotation. Torque caliper bolts to 28 Nm."
            ),
            component_tags=["brakes"], symptom_tags=[],
        ),
    ]

    # ------------------------------------------------------------ 1) index
    banner("Stage 1 — Indexing 4 chunks with REAL Gemini gemini-embedding-001")
    t0 = time.time()
    _save_chroma(chunks)
    _save_bm25(chunks)
    print(f"  ✓ indexed in {time.time() - t0:.2f}s")

    tamil_query = "என் பைக்கில் இருந்து ஏன் வெள்ளை புகை வருகிறது?"

    # ------------------------------------------------------------ 2) rewrite
    banner("Stage 2 — REAL Gemini rewriter")
    print(f"  Tamil query: {tamil_query}")
    t0 = time.time()
    rewrites, rw_usage = rewrite(tamil_query, vision_text="")
    print(f"  ✓ rewriter returned {len(rewrites)} phrases in {time.time() - t0:.2f}s")
    print(f"    tokens in/out: {rw_usage['input_tokens']}/{rw_usage['output_tokens']}")
    for r in rewrites:
        print(f"    - {r}")

    # ------------------------------------------------------------ 3) retrieve
    banner("Stage 3 — Hybrid retrieval (BM25 + multilingual vector + RRF)")
    t0 = time.time()
    retrieved = retrieval.retrieve(
        queries=rewrites, original_query=tamil_query,
        top_k=3, manual_id="re_classic_350",
    )
    print(f"  ✓ retrieved {len(retrieved)} chunks in {time.time() - t0:.2f}s")
    for rc in retrieved:
        print(f"    score={rc.score:.4f}  page {rc.chunk.page:>3}  | {rc.chunk.section}")

    # ------------------------------------------------------------ 4) sarvam
    banner("Stage 4 — REAL Sarvam 105B grounded generation")
    t0 = time.time()
    raw = generate(tamil_query, None, retrieved)
    print(f"  ✓ Sarvam returned in {time.time() - t0:.2f}s")
    print()
    print(json.dumps(raw.model_dump(), indent=2, ensure_ascii=False))

    # ------------------------------------------------------------ 5) verify
    banner("Stage 5 — Citation verifier")
    verified = verify(raw, retrieved)
    print(json.dumps(verified.model_dump(), indent=2, ensure_ascii=False))

    banner("✓ Real end-to-end test complete")


if __name__ == "__main__":
    main()
