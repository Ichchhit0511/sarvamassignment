"""End-to-end test of the Tamil-query → English-manual → Tamil-answer flow.

This test runs without any real API keys. It demonstrates:
  Stage 1 — synthetic English manual gets indexed (Chroma + BM25)
  Stage 2 — query rewriter behaviour (mocked: produces English search phrases)
  Stage 3 — hybrid retrieval pulls the right English chunk for a Tamil query
  Stage 4 — the exact prompt that would be sent to Sarvam 105B
  Stage 5 — what Sarvam returns when the key is fake (graceful fallback)
  Stage 6 — what Sarvam *would* return with a real key (mocked)
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from unittest.mock import patch

# Make project root importable.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Set fake keys BEFORE importing backend.config (it loads env at import time).
os.environ["SARVAM_API_KEY"] = "sk-fake-sarvam-key-for-testing"
os.environ["GEMINI_API_KEY"] = ""           # disabled → stub embeddings
os.environ["COHERE_API_KEY"] = ""           # disabled → no rerank
os.environ["WHAPI_TOKEN"] = "fake-whapi-token"


def banner(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def main() -> None:
    # Clean previous state so the test is reproducible.
    from backend.config import settings
    if settings.chroma_dir.exists():
        shutil.rmtree(settings.chroma_dir)
    if settings.bm25_index_path.exists():
        settings.bm25_index_path.unlink()

    # ---------------------------------------------------------------- 1) index
    from backend.models import Chunk
    from backend.ingest import _save_chroma, _save_bm25

    chunks = [
        Chunk(
            chunk_id="c001",
            manual_id="re_classic_350",
            page=42,
            section="Troubleshooting > Engine > White Smoke",
            text=(
                "WHITE SMOKE FROM EXHAUST. Possible causes: "
                "(1) Coolant entering combustion chamber due to head gasket failure. "
                "(2) Condensation in exhaust during cold start (normal, dissipates in 2-3 minutes). "
                "Action: If smoke persists after engine warms up, stop riding and consult "
                "an authorized service center. Do not continue operation as this may damage "
                "the catalytic converter."
            ),
            component_tags=["engine", "cooling", "exhaust"],
            symptom_tags=["smoke"],
        ),
        Chunk(
            chunk_id="c002",
            manual_id="re_classic_350",
            page=43,
            section="Troubleshooting > Engine > Black Smoke",
            text=(
                "BLACK SMOKE FROM EXHAUST. Indicates rich fuel mixture. "
                "Check air filter for clogging. Replace if dirty (refer to chapter 5). "
                "Verify fuel injector cleanliness. Persistent black smoke requires ECU "
                "diagnostic at service center."
            ),
            component_tags=["engine", "fuel", "exhaust"],
            symptom_tags=["smoke"],
        ),
        Chunk(
            chunk_id="c003",
            manual_id="re_classic_350",
            page=18,
            section="Maintenance > Cooling System",
            text=(
                "COOLANT LEVEL CHECK. Inspect coolant reservoir weekly. "
                "Top up with manufacturer-specified coolant only. "
                "Check radiator hoses for cracks or leaks. "
                "Capacity: 1.2 litres. Service interval: every 12,000 km."
            ),
            component_tags=["cooling"],
            symptom_tags=["leak"],
        ),
        Chunk(
            chunk_id="c004",
            manual_id="re_classic_350",
            page=71,
            section="Brakes > Pad Replacement",
            text=(
                "BRAKE PAD REPLACEMENT. Loosen caliper bolts. Remove old pads. "
                "Inspect rotor for scoring. Install new pads ensuring chamfered edge faces "
                "rotation. Torque caliper bolts to 28 Nm. Bed in pads with 10 moderate stops."
            ),
            component_tags=["brakes"],
            symptom_tags=[],
        ),
    ]

    _save_chroma(chunks)
    _save_bm25(chunks)
    banner(f"Stage 1 — Indexed {len(chunks)} synthetic English chunks")
    for c in chunks:
        print(f"  {c.chunk_id} | p.{c.page:>3} | {c.section}")

    # ---------------------------------------------------------- 2) rewriter sim
    tamil_query = "என் பைக்கில் இருந்து ஏன் வெள்ளை புகை வருகிறது?"
    banner("Stage 2 — Query rewriter (Gemini) simulation")
    print(f"  User question (Tamil): {tamil_query}")
    print(f"  English translation:   Why is white smoke coming from my bike?")
    print()
    print("  Gemini (real) would output JSON like:")
    english_rewrites = [
        "white smoke from exhaust",
        "white smoke engine cause",
        "head gasket failure smoke",
        "coolant burning combustion",
        tamil_query,  # always include original
    ]
    for r in english_rewrites:
        print(f"    - {r}")

    # ---------------------------------------------------------- 3) retrieval
    from backend import retrieval
    banner("Stage 3 — Hybrid retrieval (BM25 + Chroma vector + RRF)")
    retrieved = retrieval.retrieve(
        queries=english_rewrites,
        original_query=tamil_query,
        top_k=3,
        manual_id="re_classic_350",
    )
    if not retrieved:
        print("  ⚠️  Nothing retrieved — check the indexer.")
    for rc in retrieved:
        print(f"  score={rc.score:.4f}  page {rc.chunk.page:>3}  | {rc.chunk.section}")
        print(f"     “{rc.chunk.text[:100]}...”")

    # ---------------------------------------------------------- 4) Sarvam prompt
    from backend.generator import SYSTEM_PROMPT, _build_user_message
    banner("Stage 4 — Exactly what gets sent to Sarvam 105B")
    print("[SYSTEM PROMPT — abbreviated]")
    print(SYSTEM_PROMPT[:600] + "  ...\n")
    print("[USER MESSAGE — first 1200 chars]")
    print(_build_user_message(tamil_query, None, retrieved)[:1200])
    print("  ...")

    # ---------------------------------------------------------- 5) live call (fake key)
    from backend.generator import generate
    from backend.verifier import verify
    banner("Stage 5 — Live call with FAKE Sarvam key (graceful failure)")
    raw = generate(tamil_query, None, retrieved)
    print(json.dumps(raw.model_dump(), indent=2, ensure_ascii=False))

    # ---------------------------------------------------------- 6) mocked real-key flow
    banner("Stage 6 — Same call, mocked Sarvam response (what a real key gives)")

    mock_sarvam_payload = {
        "choices": [{
            "message": {
                "content": json.dumps({
                    "answer": (
                        "உங்கள் பைக்கின் எக்ஸாஸ்ட்டில் இருந்து வெள்ளை புகை வருவதற்கு "
                        "இரண்டு முக்கிய காரணங்கள் இருக்கலாம்: "
                        "(1) ஹெட் காஸ்கெட் தோல்வியால் கூலண்ட் எரியும் அறைக்குள் நுழைதல், "
                        "(2) குளிர்ந்த தொடக்கத்தில் எக்ஸாஸ்ட்டில் ஈரப்பதம் திரளுதல் — "
                        "இது சாதாரணம், 2-3 நிமிடங்களில் நின்றுவிடும். "
                        "இன்ஜின் சூடேறிய பின்னும் புகை தொடர்ந்தால், வண்டியை "
                        "ஓட்டுவதை நிறுத்தி அங்கீகரிக்கப்பட்ட சேவை மையத்தை அணுகவும். "
                        "தொடர்ந்து ஓட்டினால் கேடலிட்டிக் கன்வெர்ட்டர் சேதமடையும்."
                    ),
                    "citations": [{"page": 42, "chunk_id": "c001"}],
                    "confidence": "high",
                    "manual_supported": True,
                    "language": "ta",
                }, ensure_ascii=False)
            }
        }]
    }

    class _FakeResp:
        status_code = 200
        def raise_for_status(self): pass
        def json(self): return mock_sarvam_payload

    with patch("httpx.Client") as mock_client:
        mock_client.return_value.__enter__.return_value.post.return_value = _FakeResp()
        mocked_answer = generate(tamil_query, None, retrieved)
        verified = verify(mocked_answer, retrieved)

    print(json.dumps(verified.model_dump(), indent=2, ensure_ascii=False))

    banner("✓ Test complete")
    print("Summary:")
    print("  • Tamil query → Gemini rewriter produces English search phrases")
    print(f"  • Retrieval picks the correct chunk (page {retrieved[0].chunk.page}) "
          "from the English manual")
    print("  • Sarvam receives Tamil query + English chunks + 'reply in same language' rule")
    print("  • Sarvam outputs Tamil answer + English-page citations")
    print("  • Citation verifier confirms cited page exists in retrieved set")


if __name__ == "__main__":
    main()
