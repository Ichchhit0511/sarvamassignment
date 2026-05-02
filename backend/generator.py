"""Grounded generation with Sarvam (105B, multilingual).

The Sarvam model is the user-facing voice. It receives:
  - the user's original query (in any Indic or English language),
  - the visual observation (English),
  - the retrieved manual chunks (English, with page numbers),
and must answer ONLY from those chunks, in the SAME language as the query.

The system prompt enforces strict grounding. The output is JSON so we can
verify citations programmatically before showing it to the user.
"""
from __future__ import annotations

import json
import re

from .config import settings
from .models import Citation, GroundedAnswer, RetrievedChunk, VisionObservation


SYSTEM_PROMPT = """You are a bike troubleshooting assistant. Your ONLY knowledge
source is the manual chunks the user provides. Follow these rules absolutely:

1. Answer ONLY using information present in the provided manual chunks. Do not
   use general knowledge about motorcycles, even if you know the answer.
2. If the answer is not in the chunks, set "manual_supported": false and reply
   with a short refusal explaining the manual does not cover this and the user
   should consult an authorized service center.
3. Every factual claim in your answer must be traceable to one of the chunks.
   Cite chunks via the "citations" field with the page number and chunk_id.
4. Detect the language of the user's question (Hindi, Tamil, Marathi, Bengali,
   Telugu, Kannada, English, etc.) and write the "answer" field in THAT SAME
   language. Use the script of that language (Devanagari for Hindi, etc.).
   The "language" field must contain the BCP-47 code (hi, ta, mr, bn, te, kn, en).
5. Keep the answer practical and step-by-step where the manual is step-by-step.
6. Output STRICT JSON only — no markdown, no commentary outside the JSON.

Output schema:
{
  "answer": "<final answer in the user's language>",
  "citations": [{"page": <int>, "chunk_id": "<string>"}],
  "confidence": "high" | "medium" | "low",
  "manual_supported": true | false,
  "language": "<bcp-47 code>"
}
"""


def _detect_query_language(query: str) -> tuple[str, str]:
    """Lightweight language detection based on script, with English fallback."""
    for ch in query:
        cp = ord(ch)
        if 0x0900 <= cp <= 0x097F:
            return "hi", "Hindi"
        if 0x0B80 <= cp <= 0x0BFF:
            return "ta", "Tamil"
        if 0x0980 <= cp <= 0x09FF:
            return "bn", "Bengali"
        if 0x0C00 <= cp <= 0x0C7F:
            return "te", "Telugu"
        if 0x0C80 <= cp <= 0x0CFF:
            return "kn", "Kannada"
        if 0x0A80 <= cp <= 0x0AFF:
            return "gu", "Gujarati"
        if 0x0A00 <= cp <= 0x0A7F:
            return "pa", "Punjabi"
        if 0x0D00 <= cp <= 0x0D7F:
            return "ml", "Malayalam"
        if 0x0B00 <= cp <= 0x0B7F:
            return "or", "Odia"
    return "en", "English"


def _format_chunks(chunks: list[RetrievedChunk]) -> str:
    lines: list[str] = []
    for rc in chunks:
        c = rc.chunk
        lines.append(
            f"--- chunk_id={c.chunk_id} | page={c.page} | section={c.section} ---\n"
            f"{c.text}"
        )
    return "\n\n".join(lines)


def _build_user_message(query: str, vision: VisionObservation | None,
                        chunks: list[RetrievedChunk],
                        target_language_code: str,
                        target_language_name: str) -> str:
    vision_block = ""
    if vision:
        vision_block = (
            "\nVisual observation (extracted from user's photo):\n"
            f"{vision.model_dump_json()}\n"
        )

    return (
        f"CURRENT USER QUESTION LANGUAGE: {target_language_name} ({target_language_code})\n"
        "IMPORTANT: Answer in the language of the CURRENT USER QUESTION only. "
        "Do not continue the language from earlier chat history unless the current "
        "question is in that same language.\n\n"
        f"USER QUESTION: {query}\n"
        f"{vision_block}\n"
        f"MANUAL CHUNKS (your only knowledge source):\n"
        f"{_format_chunks(chunks)}\n\n"
        "Now produce the JSON response per the schema."
    )


def _parse_json(text: str) -> dict:
    text = (text or "").strip()
    fenced = re.match(r"```(?:json)?\s*(.+?)\s*```", text, re.S)
    if fenced:
        text = fenced.group(1)
    # If the model wrapped JSON in prose, try to find the outer braces.
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return {}


def _refusal(reason: str, language: str = "en") -> GroundedAnswer:
    return GroundedAnswer(
        answer=(
            "The manual does not contain specific guidance on this issue. "
            "Please consult an authorized service center."
        ),
        citations=[],
        confidence="low",
        manual_supported=False,
        language=language,
    )


def generate(query: str, vision: VisionObservation | None,
             chunks: list[RetrievedChunk],
             history: list[dict] | None = None) -> tuple[GroundedAnswer, dict]:
    """Call Sarvam 105B for the grounded answer.

    Returns:
        (GroundedAnswer, usage)  where usage = {input_tokens, output_tokens}
    """
    usage = {"input_tokens": 0, "output_tokens": 0}
    target_language_code, target_language_name = _detect_query_language(query)

    if not chunks:
        return _refusal("No manual chunks retrieved.", language=target_language_code), usage

    if not settings.sarvam_api_key:
        return GroundedAnswer(
            answer=(
                "(Sarvam API key not configured. Add SARVAM_API_KEY to api.env "
                "to enable multilingual answers. Top retrieved chunk preview: "
                f"{chunks[0].chunk.text[:200]}...)"
            ),
            citations=[Citation(page=chunks[0].chunk.page, chunk_id=chunks[0].chunk.chunk_id)],
            confidence="low",
            manual_supported=False,
            language=target_language_code,
        ), usage

    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({
        "role": "user",
        "content": _build_user_message(
            query,
            vision,
            chunks,
            target_language_code,
            target_language_name,
        ),
    })

    try:
        from sarvamai import SarvamAI
        client = SarvamAI(api_subscription_key=settings.sarvam_api_key)
        response = client.chat.completions(
            model=settings.sarvam_model,
            messages=messages,
            temperature=0.2,
            top_p=1,
            max_tokens=2000,
        )
    except Exception as e:
        return GroundedAnswer(
            answer=f"(Sarvam API error: {e}. Check api.env and try again.)",
            citations=[],
            confidence="low",
            manual_supported=False,
            language=target_language_code,
        ), usage

    # Token usage — the SDK response mirrors OpenAI's shape but field access varies.
    u = getattr(response, "usage", None)
    if u is not None:
        if hasattr(u, "prompt_tokens"):
            usage["input_tokens"] = int(u.prompt_tokens or 0)
            usage["output_tokens"] = int(getattr(u, "completion_tokens", 0) or 0)
        elif isinstance(u, dict):
            usage["input_tokens"] = int(u.get("prompt_tokens") or 0)
            usage["output_tokens"] = int(u.get("completion_tokens") or 0)

    try:
        raw = response.choices[0].message.content or ""
    except Exception:
        raw = str(response)
    parsed = _parse_json(raw)

    citations = [
        Citation(page=int(c.get("page", 0)), chunk_id=str(c.get("chunk_id", "")))
        for c in parsed.get("citations", [])
        if isinstance(c, dict)
    ]
    return GroundedAnswer(
        answer=parsed.get("answer") or raw or "",
        citations=citations,
        confidence=parsed.get("confidence", "low"),
        manual_supported=bool(parsed.get("manual_supported", False)),
        language=parsed.get("language") or target_language_code,
    ), usage
