"""FastAPI backend.

Endpoints:
  GET  /                         → web UI
  POST /api/ingest               → upload + ingest a PDF manual
  GET  /api/manuals              → list ingested manuals
  POST /api/query                → run the troubleshooting pipeline (UI calls this)
  POST /whatsapp/webhook         → Whapi inbound message webhook
  GET  /api/metrics              → dashboard aggregates
  GET  /api/metrics/recent       → last N queries
  POST /api/memory/clear         → clear a session's memory
"""
from __future__ import annotations

import logging
import re
import shutil
import time
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import memory, metrics, whatsapp
from .config import PROJECT_ROOT, settings
from .generator import generate, sanitize_answer_text
from .ingest import ingest_pdf
from .models import (
    GroundedAnswer,
    QueryMetrics,
    QueryRequest,
    QueryResponse,
)
from .query_rewriter import rewrite
from .retrieval import _load_bm25, retrieve
from .verifier import verify
from .vision import describe_image, vision_to_text


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("bike-bot")

app = FastAPI(title="Bike Troubleshooting Bot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = PROJECT_ROOT / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ---------------------------------------------------------------------------
# Web UI
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    return JSONResponse({"ok": True, "msg": "Backend running. UI not built."})


# ---------------------------------------------------------------------------
# Manuals
# ---------------------------------------------------------------------------
@app.post("/api/ingest")
async def ingest(file: UploadFile = File(...), manual_id: str = Form(...)):
    settings.manuals_dir.mkdir(parents=True, exist_ok=True)
    target = settings.manuals_dir / f"{manual_id}.pdf"
    with open(target, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        result = ingest_pdf(target, manual_id)
    except Exception as e:
        raise HTTPException(500, f"Ingestion failed: {e}")
    return {"ok": True, **result}


@app.get("/api/manuals")
def list_manuals():
    data = _load_bm25()
    if not data:
        return {"manuals": []}
    seen: dict[str, int] = {}
    for c in data.get("chunks", []):
        seen[c["manual_id"]] = seen.get(c["manual_id"], 0) + 1
    return {"manuals": [
        {"manual_id": mid, "chunk_count": n} for mid, n in seen.items()
    ]}


# ---------------------------------------------------------------------------
# Pipeline — instrumented + memory-aware
# ---------------------------------------------------------------------------
def _ms_since(t0: float) -> int:
    return int((time.time() - t0) * 1000)


def _normalize_query(query: str) -> str:
    q = (query or "").strip().lower()
    q = re.sub(r"[^\w\s]", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q


def _detect_query_language(query: str) -> str:
    q = _normalize_query(query)
    if q in {"namaste", "namaskar", "namaskaar"}:
        return "hi"
    roman_hindi_hints = {
        "mera", "meri", "mere", "mujhe", "bike", "problem", "dikkat",
        "takleef", "hai", "ho", "gayi", "gyi", "kya", "kaise", "nahi",
        "issue", "me", "mein",
    }
    if sum(1 for w in q.split() if w in roman_hindi_hints) >= 2:
        return "hi-Latn"
    for ch in query:
        cp = ord(ch)
        if 0x0900 <= cp <= 0x097F:
            return "hi"
        if 0x0A80 <= cp <= 0x0AFF:
            return "gu"
        if 0x0B80 <= cp <= 0x0BFF:
            return "ta"
    return "en"


def _heuristic_query_category(query: str, image_b64: Optional[str]) -> str:
    if image_b64:
        return "bike"
    q = _normalize_query(query)
    if not q:
        return "greeting"
    if query.strip() in {"வணக்கம்", "नमस्ते", "नमस्कार"}:
        return "greeting"
    greeting_patterns = [
        r"^(hi|hello|hey|hii|good morning|good afternoon|good evening)$",
        r"^(namaste|namaskar|namaskaar)$",
        r"^(who are you|what can you do|help|start)$",
        r"^(thanks|thank you|ok|okay)$",
    ]
    if any(re.match(pattern, q) for pattern in greeting_patterns):
        return "greeting"
    return "bike"


def _translate_query_to_english(query: str) -> str:
    query = (query or "").strip()
    if not query:
        return ""
    if _detect_query_language(query) == "en":
        return query
    if not settings.gemini_api_key:
        return query

    try:
        import google.generativeai as genai

        genai.configure(api_key=settings.gemini_api_key)
        model = genai.GenerativeModel(settings.gemini_rewriter_model)
        prompt = (
            "Translate the following user query into natural English.\n"
            "Keep the meaning exact.\n"
            "Return only the translated English text, with no explanation.\n\n"
            f"Query: {query}"
        )
        resp = model.generate_content(prompt)
        text = (getattr(resp, "text", "") or "").strip()
        if text:
            return text
    except Exception:
        pass
    return query


def _sarvam_greeting_decision(english_query: str) -> Optional[str]:
    english_query = (english_query or "").strip()
    if not english_query or not settings.sarvam_api_key:
        return None

    messages = [
        {
            "role": "system",
            "content": (
                "You classify whether a user message is ONLY a simple greeting or basic social opener.\n"
                "Think step by step if needed, but end with a final line exactly like RESULT=GREETING or RESULT=BIKE.\n"
                "Return GREETING only when the message is just a greeting/opening such as "
                "'hi', 'hello', 'good morning', 'namaste', 'thanks', 'who are you', or 'help'.\n"
                "If the message contains any bike issue, symptom, manual/specification request, "
                "question, image-related request, or any meaningful content beyond a simple greeting, "
                "return BIKE."
            ),
        },
        {"role": "user", "content": english_query},
    ]

    try:
        from sarvamai import SarvamAI

        client = SarvamAI(api_subscription_key=settings.sarvam_api_key)
        response = client.chat.completions(
            model=settings.sarvam_model,
            messages=messages,
            temperature=0,
            top_p=1,
            max_tokens=256,
        )
        msg = response.choices[0].message
        raw = (
            getattr(msg, "content", None)
            or getattr(msg, "reasoning_content", None)
            or ""
        ).strip().upper()
        if "RESULT=GREETING" in raw:
            return "greeting"
        if "RESULT=BIKE" in raw:
            return "bike"
        if "GREETING" in raw:
            return "greeting"
        if "BIKE" in raw:
            return "bike"
    except Exception:
        pass
    return None


def _classify_query_category(query: str, image_b64: Optional[str]) -> str:
    heuristic = _heuristic_query_category(query, image_b64)
    if image_b64:
        return heuristic

    english_query = _translate_query_to_english(query)
    llm_decision = _sarvam_greeting_decision(english_query)
    english_heuristic = _heuristic_query_category(english_query, None)
    if llm_decision == "greeting" and english_heuristic != "greeting":
        return "bike"
    return llm_decision or heuristic


def _general_chat_answer(query: str) -> GroundedAnswer:
    lang = _detect_query_language(query)
    if lang == "hi":
        text = (
            "नमस्ते, मैं Royal Enfield bike assistant हूं। आप अपनी बाइक में आ रही किसी भी "
            "समस्या के बारे में पूछ सकते हैं, या बाइक की मुख्य technical specifications जान सकते हैं।"
        )
    elif lang == "ta":
        text = (
            "வணக்கம், நான் Royal Enfield bike assistant. உங்கள் பைக்கில் இருக்கும் "
            "எந்த பிரச்சனையையும் கேட்கலாம், அல்லது பைக்கின் முக்கிய technical specifications-ஐ கேட்கலாம்."
        )
    else:
        text = (
            "Hi, I am the Royal Enfield bike assistant. You can ask me about any "
            "issue you are facing with the bike, or about key technical specifications of the bike."
        )
    return GroundedAnswer(
        answer=text,
        citations=[],
        confidence="high",
        manual_supported=True,
        language=lang,
    )


def _localized_manual_refusal(language: str) -> str:
    if language == "hi-Latn":
        return (
            "Manual mein is issue ke baare mein paryapt aur clear jaankari nahi hai. "
            "Kripya authorized service center se sampark karein ya nearest service centre par jaayein."
        )
    if language == "hi":
        return (
            "मैनुअल में इस समस्या के बारे में पर्याप्त और स्पष्ट जानकारी उपलब्ध नहीं है। "
            "कृपया अधिकृत सर्विस सेंटर से संपर्क करें या अपने नजदीकी सर्विस सेंटर पर जाएं।"
        )
    if language == "gu":
        return (
            "મેન્યુઅલમાં આ સમસ્યા વિશે પૂરતી અને સ્પષ્ટ માહિતી ઉપલબ્ધ નથી. "
            "કૃપા કરીને અધિકૃત સર્વિસ સેન્ટરનો સંપર્ક કરો અથવા નજીકના સર્વિસ સેન્ટર પર જાઓ."
        )
    if language == "ta":
        return (
            "இந்த பிரச்சனை குறித்து கையேட்டில் போதுமான தெளிவான தகவல் இல்லை. "
            "அதிகாரப்பூர்வ சேவை மையத்தை தொடர்புகொள்ளவும் அல்லது அருகிலுள்ள சேவை மையத்திற்குச் செல்லவும்."
        )
    return (
        "The manual does not contain enough clear information about this issue. "
        "Please contact an authorized service center or visit the nearest service centre."
    )


def _is_generic_image_diagnosis_query(query: str, image_b64: Optional[str]) -> bool:
    if not image_b64:
        return False
    q = _normalize_query(query)
    patterns = [
        r"\bimage\b",
        r"\bphoto\b",
        r"\bpicture\b",
        r"\bthis image\b",
        r"\bthis photo\b",
        r"\bwhat issue\b",
        r"\bidentify\b",
        r"\bvisible\b",
        r"क्या समस्या",
        r"फोटो",
        r"तस्वीर",
    ]
    return any(re.search(pattern, q) for pattern in patterns)


def _image_issue_supported_by_chunks(vision_text: str, retrieved) -> bool:
    vt = (vision_text or "").lower()
    if "leak" in vt:
        keywords = ("leak", "leaking", "drip", "seep", "oil seal", "gasket")
    elif "smoke" in vt:
        keywords = ("smoke", "smoking", "fume", "exhaust")
    elif "damage" in vt or "wear" in vt:
        keywords = ("damage", "wear", "crack", "replace", "inspect")
    else:
        keywords = ()
    if not keywords:
        return True
    for rc in retrieved:
        text = rc.chunk.text.lower()
        if any(k in text for k in keywords):
            return True
    return False


def _run_pipeline(query: str, image_b64: Optional[str],
                  manual_id: Optional[str],
                  session_id: Optional[str] = None,
                  answer_model: str = "sarvam") -> QueryResponse:
    overall_t0 = time.time()
    error_str: Optional[str] = None

    # Stage 1: Vision
    t0 = time.time()
    vision = None
    vision_usage = {"input_tokens": 0, "output_tokens": 0}
    if image_b64:
        vision, vision_usage = describe_image(image_b64)
    vision_text = vision_to_text(vision)
    vision_ms = _ms_since(t0)

    # Stage 2: Rewrite
    t0 = time.time()
    rewrites, rewrite_usage = rewrite(query, vision_text)
    rewrite_ms = _ms_since(t0)
    log.info("Rewrites: %s", rewrites)

    # Stage 3: Retrieve
    t0 = time.time()
    retrieved = retrieve(
        queries=rewrites,
        original_query=query,
        top_k=5,
        manual_id=manual_id,
    )
    retrieve_ms = _ms_since(t0)
    log.info("Retrieved %d chunks", len(retrieved))
    top_score = retrieved[0].score if retrieved else 0.0

    # Stage 4: Generate (with conversation history)
    history = memory.get_history(session_id) if session_id else []
    t0 = time.time()
    raw_answer, sarvam_usage = generate(
        query,
        vision,
        retrieved,
        history=history,
        answer_model=answer_model,
    )
    generate_ms = _ms_since(t0)

    # Stage 5: Verify
    t0 = time.time()
    answer = verify(raw_answer, retrieved)
    answer.answer = sanitize_answer_text(answer.answer)
    if (
        answer.manual_supported
        and _is_generic_image_diagnosis_query(query, image_b64)
        and not _image_issue_supported_by_chunks(vision_text, retrieved)
    ):
        answer = GroundedAnswer(
            answer=_localized_manual_refusal(answer.language or _detect_query_language(query)),
            citations=[],
            confidence="low",
            manual_supported=False,
            language=answer.language or _detect_query_language(query),
        )
    verify_ms = _ms_since(t0)

    total_ms = _ms_since(overall_t0)

    # Persist conversation turn (for follow-up questions).
    if session_id:
        memory.append_turn(session_id, "user", query, language=None)
        if answer.answer:
            memory.append_turn(session_id, "assistant", answer.answer,
                               language=answer.language)

    # Log metrics row.
    gemini_in = vision_usage["input_tokens"] + rewrite_usage["input_tokens"]
    gemini_out = vision_usage["output_tokens"] + rewrite_usage["output_tokens"]
    metrics.record(
        session_id=session_id,
        manual_id=manual_id,
        query=query[:500],
        language=answer.language,
        has_image=1 if image_b64 else 0,
        num_rewrites=len(rewrites),
        num_retrieved=len(retrieved),
        top_retrieval_score=top_score,
        manual_supported=1 if answer.manual_supported else 0,
        confidence=answer.confidence,
        num_citations_raw=len(raw_answer.citations),
        num_citations_kept=len(answer.citations),
        sarvam_input_tokens=sarvam_usage["input_tokens"],
        sarvam_output_tokens=sarvam_usage["output_tokens"],
        gemini_input_tokens=gemini_in,
        gemini_output_tokens=gemini_out,
        vision_ms=vision_ms,
        rewrite_ms=rewrite_ms,
        retrieve_ms=retrieve_ms,
        generate_ms=generate_ms,
        verify_ms=verify_ms,
        total_ms=total_ms,
        error=error_str,
    )

    return QueryResponse(
        answer=answer,
        vision=vision,
        rewrites=rewrites,
        retrieved=retrieved,
        metrics=QueryMetrics(
            total_ms=total_ms,
            vision_ms=vision_ms,
            rewrite_ms=rewrite_ms,
            retrieve_ms=retrieve_ms,
            generate_ms=generate_ms,
            verify_ms=verify_ms,
            sarvam_input_tokens=sarvam_usage["input_tokens"],
            sarvam_output_tokens=sarvam_usage["output_tokens"],
            gemini_input_tokens=gemini_in,
            gemini_output_tokens=gemini_out,
            top_retrieval_score=top_score,
            num_citations_kept=len(answer.citations),
        ),
    )


@app.post("/api/query", response_model=QueryResponse)
async def api_query(req: QueryRequest):
    try:
        return _run_pipeline(
            req.query,
            req.image_b64,
            req.manual_id,
            req.session_id,
            req.answer_model or "sarvam",
        )
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Query pipeline error")
        raise HTTPException(500, f"Query failed: {e}")


# ---------------------------------------------------------------------------
# Memory management
# ---------------------------------------------------------------------------
@app.post("/api/memory/clear")
async def memory_clear(req: Request):
    body = await req.json()
    sid = (body.get("session_id") or "").strip()
    if not sid:
        return {"ok": False, "error": "Missing session_id"}
    n = memory.clear_session(sid)
    return {"ok": True, "deleted": n}


# ---------------------------------------------------------------------------
# Dashboard / metrics endpoints
# ---------------------------------------------------------------------------
@app.get("/api/metrics")
def get_metrics(window_hours: int = 168):
    return metrics.aggregate(window_hours=window_hours)


@app.get("/api/metrics/recent")
def get_recent(limit: int = 30):
    return {"queries": metrics.recent(limit=limit)}


# ---------------------------------------------------------------------------
# Whapi webhook
# ---------------------------------------------------------------------------
def _format_for_whatsapp(answer: GroundedAnswer) -> str:
    body = answer.answer.strip() or "(empty answer)"
    if answer.manual_supported and answer.citations:
        pages = sorted({c.page for c in answer.citations if c.page})
        if len(pages) == 1:
            body += f"\n\n📖 You can refer to page {pages[0]} of the manual for more details."
        elif pages:
            body += f"\n\n📖 You can refer to pages {', '.join(map(str, pages))} of the manual for more details."
    return body


@app.post("/api/whatsapp/send")
async def whatsapp_send(req: Request):
    body = await req.json()
    to = (body.get("to") or "").strip()
    text = (body.get("body") or "").strip()
    if not to or not text:
        return {"ok": False, "error": "Missing 'to' or 'body'"}
    return whatsapp.send_text(to, text)


@app.post("/whatsapp/webhook")
async def whatsapp_webhook(request: Request):
    payload = await request.json()
    log.info("Whapi inbound: %s", str(payload)[:500])
    parsed = whatsapp.parse_inbound(payload)
    sender = parsed["sender"]
    text = parsed["text"] or ""
    image_b64 = parsed["image_b64"]

    if not sender or (not text and not image_b64):
        return {"ok": True, "skipped": True}

    user_query = text or "What issue does this image show? Help me troubleshoot."

    manuals = list_manuals().get("manuals", [])
    manual_id = manuals[0]["manual_id"] if manuals else None

    if not manual_id:
        whatsapp.send_text(sender, "No manual is loaded yet. Please ask the bot owner to ingest a manual.")
        return {"ok": True, "no_manual": True}

    try:
        # WhatsApp sender phone is the natural session_id → memory persists across messages.
        result = _run_pipeline(user_query, image_b64, manual_id, session_id=sender)
        whatsapp.send_text(sender, _format_for_whatsapp(result.answer))
    except Exception as e:
        log.exception("Pipeline error")
        whatsapp.send_text(sender, f"Sorry, an error occurred: {e}")
    return {"ok": True}


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {
        "ok": True,
        "sarvam_key_set": bool(settings.sarvam_api_key),
        "gemini_key_set": bool(settings.gemini_api_key),
        "cohere_key_set": bool(settings.cohere_api_key),
        "whapi_token_set": bool(settings.whapi_token),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host=settings.host, port=settings.port, reload=True)
