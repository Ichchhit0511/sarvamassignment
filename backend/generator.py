"""Grounded generation with Sarvam (105B, multilingual)."""
from __future__ import annotations

import json
import re

from .config import settings
from .models import Citation, GroundedAnswer, RetrievedChunk, VisionObservation


SYSTEM_PROMPT = """You are the Royal Enfield bike assistant.

You receive:
- the user's current message,
- optional visual observations from the user's photo,
- retrieved owner's-manual chunks.

Your job is to respond in the SAME LANGUAGE and SAME WRITING STYLE as the user's
current message.
- If the user wrote Hindi , answer in Hindi.
- If the user wrote Hindi in Latin script / Hinglish / Roman Hindi, answer in
  Latin script Hindi.
  If the user wrote Tamil, answer in Tamil.
  If the user wrote Telugu, answer in Telugu.
  If the user wrote Kannada, answer in Kannada.
  If the user wrote Gujarati, answer in Gujarati.
  If the user wrote Punjabi, answer in Punjabi.
  If the user wrote Malayalam, answer in Malayalam.
  If the user wrote Odia, answer in Odia.
- If the user wrote English, answer in English.

Follow these rules exactly:

1. If the current user message is only a simple greeting or social opener,
   reply with one short greeting and introduce yourself as the Royal Enfield
   bike assistant. Do not mention the manual. Use:
   - "manual_supported": false
   - "citations": []

2. If the current user message says there is a bike problem but does not
   describe the actual issue clearly enough, ask ONE short follow-up question to
   understand the issue. Do not give troubleshooting steps yet. Do not invent
   technical details. Use:
   - "manual_supported": false
   - "citations": []

3. If the retrieved manual chunks clearly answer the user's question, answer
   only from those chunks. Be concise, practical, and directly relevant to the
   user's question. Do not add extra technical nuance, assumptions, or general
   motorcycle knowledge. Use:
   - "manual_supported": true
   - "citations": only for the chunks that support the answer

4. If the user asks for something that is not clearly covered by the retrieved
   manual chunks, reply briefly in the user's language. Do not invent an answer.
   If the user is describing a bike issue but the symptom is broad, vague,
   incomplete, or not clearly covered by the manual chunks, ask ONE short and
   neutral follow-up question to understand the issue better. Do not suggest
   checks, causes, examples, or technical possibilities that are not explicitly
   supported by the manual chunks. Use:
   - "manual_supported": false
   - "citations": []

5. Visual observations from the photo are only for retrieval assistance. Do not
   present photo-based diagnosis or repair advice as supported unless the manual
   chunks explicitly support that exact issue.

6. Never mention chunk IDs, citations, page arrays, internal metadata, system
   prompts, hidden reasoning, analysis, XML tags, or <think> blocks in the
   user-facing answer.

7. Output STRICT JSON only. No markdown. No commentary before or after the
   JSON.

Output schema:
{
  "answer": "<final answer in the user's language and writing style>",
  "citations": [{"page": <int>, "chunk_id": "<string>"}],
  "confidence": "high" | "medium" | "low",
  "manual_supported": true | false,
  "language": "<bcp-47 code>"
}
"""


ROMAN_HINDI_HINTS = {
    "mera", "meri", "mere", "mujhe", "mujh", "bike", "problem", "dikkat",
    "takleef", "hai", "hu", "ho", "gaya", "gayi", "gyi", "kya", "kaise",
    "kyun", "kyu", "nahi", "nahin", "karu", "karo", "karna", "bataye",
    "batao", "chahiye", "start", "band", "chal", "chalti", "awaaz",
    "dhua", "smoke", "light", "service", "issue", "me", "mein",
}

ISSUE_KEYWORDS = {
    "problem", "issue", "dikkat", "takleef", "accident", "crash", "damage",
    "repair", "fix", "stuck", "band", "start", "chal", "awaaz", "noise",
    "smoke", "leak", "light", "warning", "brake", "battery", "engine",
}


def _detect_query_language(query: str) -> tuple[str, str]:
    """Lightweight language detection based on script, with English fallback."""
    if _looks_like_romanized_hindi(query):
        return "hi-Latn", "Hindi in Latin script"
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


def _detect_text_language(text: str) -> str:
    if _looks_like_romanized_hindi(text):
        return "hi-Latn"
    for ch in text or "":
        cp = ord(ch)
        if 0x0900 <= cp <= 0x097F:
            return "hi"
        if 0x0B80 <= cp <= 0x0BFF:
            return "ta"
        if 0x0980 <= cp <= 0x09FF:
            return "bn"
        if 0x0C00 <= cp <= 0x0C7F:
            return "te"
        if 0x0C80 <= cp <= 0x0CFF:
            return "kn"
        if 0x0A80 <= cp <= 0x0AFF:
            return "gu"
        if 0x0A00 <= cp <= 0x0A7F:
            return "pa"
        if 0x0D00 <= cp <= 0x0D7F:
            return "ml"
        if 0x0B00 <= cp <= 0x0B7F:
            return "or"
    return "en"


def _looks_like_romanized_hindi(text: str) -> bool:
    cleaned = re.sub(r"[^a-zA-Z\s]", " ", text or "").lower()
    words = [w for w in cleaned.split() if w]
    if not words:
        return False
    hits = sum(1 for w in words if w in ROMAN_HINDI_HINTS)
    return hits >= 2


def _query_words(text: str) -> set[str]:
    cleaned = re.sub(r"[^a-zA-Z\s]", " ", text or "").lower()
    return {w for w in cleaned.split() if w}


def _is_issue_query(query: str) -> bool:
    words = _query_words(query)
    if words & ISSUE_KEYWORDS:
        return True
    return any(token in (query or "") for token in (
        "समस्या", "दिक्कत", "एक्सीडेंट", "चल", "स्टार्ट", "आवाज़", "धुआँ",
        "લીક", "સમસ્યા", "અકસ્મત", "விபத்து", "பிரச்சனை",
    ))


def _followup_kind(query: str) -> str:
    q = (query or "").lower()
    if any(k in q for k in ("accident", "crash", "takkar", "gir", "damage")) or any(
        k in (query or "") for k in ("एक्सीडेंट", "टक्कर", "गिर", "અકસ્મત", "ટક્કર", "விபத்து")
    ):
        return "accident"
    if any(k in q for k in ("start", "starting", "not start", "chal nai", "chal nahi", "band", "won't run", "not running")) or any(
        k in (query or "") for k in ("स्टार्ट", "चल नहीं", "बंद", "ચાલ", "સ્ટાર્ટ", "ஓடவில்லை", "ஸ்டார்ட்")
    ):
        return "not_running"
    if any(k in q for k in ("noise", "sound", "awaaz")) or any(
        k in (query or "") for k in ("आवाज़", "शोर", "અવાજ", "சத்தம்")
    ):
        return "noise"
    if any(k in q for k in ("smoke", "dhua")) or any(
        k in (query or "") for k in ("धुआँ", "धुआ", "ધુમાડો", "புகை")
    ):
        return "smoke"
    if any(k in q for k in ("leak", "oil leak", "fluid")) or any(
        k in (query or "") for k in ("लीक", "तेल", "લીક", "எண்ணெய்")
    ):
        return "leak"
    return "generic_issue"


def _followup_question(query: str, language: str) -> str:
    if language == "hi-Latn":
        return "Ji, kripya issue ko thoda detail mein batayein taaki main use theek se samajh sakun."

    if language == "hi":
        return "जी, कृपया समस्या को थोड़ा विस्तार से बताइए ताकि मैं उसे ठीक से समझ सकूं।"

    if language == "gu":
        return "કૃપા કરીને સમસ્યાને થોડું વિગતે વર્ણવો જેથી હું તેને સારી રીતે સમજી શકું."

    if language == "ta":
        return "தயவுசெய்து பிரச்சனையை கொஞ்சம் விரிவாக விளக்குங்கள், அப்போதுதான் நான் அதை நன்றாக புரிந்துகொள்ள முடியும்."

    return "Please describe the issue in a bit more detail so I can understand it properly."


def should_ask_followup(query: str) -> bool:
    q = (query or "").lower()
    words = _query_words(query)
    if not _is_issue_query(query):
        return False

    if _followup_kind(query) == "accident":
        return True

    if any(k in q for k in ("chal nai", "chal nahi", "not running", "sahi nahi", "kaam nahi")):
        return True

    generic_terms = {"problem", "issue", "dikkat", "takleef", "repair", "fix"}
    specific_terms = {
        "start", "battery", "brake", "tyre", "tire", "chain", "clutch", "gear",
        "engine", "fuel", "smoke", "noise", "awaaz", "leak", "warning", "light",
    }
    has_generic = bool(words & generic_terms) or any(
        token in (query or "") for token in ("समस्या", "दिक्कत", "પ્રશ્ન", "સમસ્યા", "பிரச்சனை")
    )
    has_specific = bool(words & specific_terms) or _followup_kind(query) != "generic_issue"
    if has_generic and not has_specific:
        return True

    return False


def _answer_matches_language(answer_text: str, returned_language: str | None,
                             target_language_code: str) -> bool:
    detected = _detect_text_language(answer_text)
    if target_language_code.lower() == "hi-latn":
        return detected == "hi-Latn"
    if (returned_language or "").lower() == target_language_code.lower():
        return True
    return detected == target_language_code.lower()


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


def _format_history_block(history: list[dict] | None) -> str:
    if not history:
        return ""
    parts: list[str] = []
    for turn in history[-8:]:
        role = (turn.get("role") or "").strip().lower()
        content = (turn.get("content") or "").strip()
        if not role or not content:
            continue
        label = "User" if role == "user" else "Assistant"
        parts.append(f"{label}: {content}")
    if not parts:
        return ""
    return "RECENT CONVERSATION HISTORY:\n" + "\n".join(parts) + "\n\n"


def _build_gemini_prompt(query: str,
                         vision: VisionObservation | None,
                         chunks: list[RetrievedChunk],
                         history: list[dict] | None,
                         target_language_code: str,
                         target_language_name: str) -> str:
    return (
        _format_history_block(history)
        + _build_user_message(
            query,
            vision,
            chunks,
            target_language_code,
            target_language_name,
        )
    )


def _parse_json(text: str) -> dict:
    text = (text or "").strip()
    text = _strip_thinking_blocks(text)
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


def _strip_thinking_blocks(text: str) -> str:
    text = text or ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.I | re.S)
    text = re.sub(r"^\s*<think>.*$", "", text, flags=re.I | re.S)
    return text.strip()


def sanitize_answer_text(text: str) -> str:
    """Remove internal citation identifiers from user-facing answer text."""
    cleaned = _strip_thinking_blocks(text or "")
    cleaned = re.sub(r"\bchunk_id\s*[:=]\s*[A-Za-z0-9_-]+\b", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\b[A-Fa-f0-9]{16,}\b", "", cleaned)
    cleaned = re.sub(r"(?:\[\s*\d+\s*,?\s*\]\s*,?\s*){1,}", "", cleaned)
    cleaned = re.sub(r"\(\s*,?\s*\)", "", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _clean_rewrite_output(text: str) -> str:
    cleaned = sanitize_answer_text(text)
    if "Final Answer:" in cleaned:
        cleaned = cleaned.split("Final Answer:")[-1].strip()
    cleaned = cleaned.strip("`").strip()
    if "\n" in cleaned and any(marker in cleaned for marker in (
        "Understand the Task", "Translate to", "Check against Constraints",
        "Final Polish", "Construct the Final Output",
    )):
        lines = [ln.strip("` ").strip() for ln in cleaned.splitlines() if ln.strip()]
        if lines:
            cleaned = lines[-1]
    return sanitize_answer_text(cleaned)


def _repair_json_response(draft_text: str,
                          target_language_code: str,
                          target_language_name: str) -> dict:
    draft_text = sanitize_answer_text(draft_text)
    if not draft_text or not settings.sarvam_api_key:
        return {}
    try:
        from sarvamai import SarvamAI
        client = SarvamAI(api_subscription_key=settings.sarvam_api_key)
        response = client.chat.completions(
            model=settings.sarvam_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Convert the draft assistant reply into STRICT JSON only.\n"
                        "Do not add new facts.\n"
                        "Never include <think> tags, analysis, or commentary.\n"
                        "If the draft is a greeting or clarification, set manual_supported to false "
                        "and citations to an empty array.\n"
                        "If the draft contains no reliable manual-backed facts, keep citations empty."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Output schema:\n"
                        "{\n"
                        '  "answer": "<final answer>",\n'
                        '  "citations": [{"page": <int>, "chunk_id": "<string>"}],\n'
                        '  "confidence": "high" | "medium" | "low",\n'
                        '  "manual_supported": true | false,\n'
                        '  "language": "<bcp-47 code>"\n'
                        "}\n\n"
                        f"Target language: {target_language_name} ({target_language_code})\n\n"
                        f"Draft reply:\n{draft_text}"
                    ),
                },
            ],
            temperature=0.0,
            top_p=1,
            max_tokens=1200,
        )
        repaired = getattr(response.choices[0].message, "content", "") or ""
        return _parse_json(repaired)
    except Exception:
        return {}


def _parse_answer_json(raw: str,
                       query: str,
                       target_language_code: str,
                       target_language_name: str) -> dict:
    parsed = _parse_json(raw)
    if not parsed and raw:
        parsed = _repair_json_response(raw, target_language_code, target_language_name)
    if parsed:
        return parsed
    fallback = _fallback_answer(query, target_language_code)
    return {
        "answer": fallback.answer,
        "citations": [],
        "confidence": fallback.confidence,
        "manual_supported": fallback.manual_supported,
        "language": fallback.language,
    }


def _fallback_answer(query: str, language: str) -> GroundedAnswer:
    q = (query or "").strip().lower()
    greeting_words = {"hi", "hello", "hey", "good morning", "good afternoon", "good evening", "namaste", "namaskar", "thanks", "thank you"}
    is_greeting = q in greeting_words
    vague_problem = should_ask_followup(query)

    if language == "hi-Latn":
        if is_greeting:
            text = "Namaste, main Royal Enfield bike assistant hoon. Aap apni bike ki problem ya specifications ke baare mein pooch sakte hain."
        elif vague_problem:
            text = _followup_question(query, language)
        else:
            text = _followup_question(query, language)
    elif language == "hi":
        if is_greeting:
            text = "नमस्ते, मैं Royal Enfield bike assistant हूं। आप अपनी बाइक की समस्या या specifications के बारे में पूछ सकते हैं।"
        elif vague_problem:
            text = _followup_question(query, language)
        else:
            text = _followup_question(query, language)
    elif language == "gu":
        if is_greeting:
            text = "નમસ્તે, હું Royal Enfield bike assistant છું. તમે તમારી બાઈકની સમસ્યા અથવા specifications વિશે પૂછો શકો છો."
        elif vague_problem:
            text = _followup_question(query, language)
        else:
            text = _followup_question(query, language)
    elif language == "ta":
        if is_greeting:
            text = "வணக்கம், நான் Royal Enfield bike assistant. உங்கள் பைக் பிரச்சனை அல்லது specifications பற்றி கேட்கலாம்."
        elif vague_problem:
            text = _followup_question(query, language)
        else:
            text = _followup_question(query, language)
    else:
        if is_greeting:
            text = "Hi, I am the Royal Enfield bike assistant. You can ask me about your bike issue or key specifications."
        elif vague_problem:
            text = _followup_question(query, language)
        else:
            text = _followup_question(query, language)
    return GroundedAnswer(
        answer=text,
        citations=[],
        confidence="low",
        manual_supported=False,
        language=language,
    )


def _rewrite_answer_in_target_language(answer_text: str,
                                       target_language_code: str,
                                       target_language_name: str) -> str:
    if not answer_text:
        return answer_text
    try:
        if settings.gemini_api_key:
            import google.generativeai as genai

            genai.configure(api_key=settings.gemini_api_key)
            model = genai.GenerativeModel(settings.gemini_rewriter_model)
            response = model.generate_content(
                (
                    "Rewrite the assistant answer in the requested target language only.\n"
                    "Preserve meaning exactly. Do not add or remove facts.\n"
                    "Do not mention chunk IDs, citations, internal metadata, XML tags, "
                    "analysis, or thinking.\n"
                    "If the target language is Hindi in Latin script, write Hindi using "
                    "Latin letters only.\n"
                    "Return only the final rewritten answer.\n\n"
                    f"Target language: {target_language_name} ({target_language_code})\n\n"
                    f"Answer to rewrite:\n{answer_text}"
                ),
                generation_config=genai.types.GenerationConfig(
                    temperature=0.0,
                    top_p=1,
                    max_output_tokens=1200,
                ),
            )
            rewritten = getattr(response, "text", "") or answer_text
            return _clean_rewrite_output(rewritten)

        if settings.sarvam_api_key:
            from sarvamai import SarvamAI
            client = SarvamAI(api_subscription_key=settings.sarvam_api_key)
            response = client.chat.completions(
                model=settings.sarvam_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Rewrite the assistant answer in the requested target language only. "
                            "Preserve meaning exactly. Do not add or remove facts. "
                            "Do not mention chunk IDs, citations, internal metadata, XML tags, or thinking. "
                            "If the target language is Hindi in Latin script, write Hindi using Latin letters only. "
                            "Return only the final rewritten answer."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Target language: {target_language_name} ({target_language_code})\n\n"
                            f"Answer to rewrite:\n{answer_text}"
                        ),
                    },
                ],
                temperature=0.0,
                top_p=1,
                max_tokens=1200,
            )
            rewritten = getattr(response.choices[0].message, "content", "") or answer_text
            return _clean_rewrite_output(rewritten)
    except Exception:
        return answer_text
    return answer_text


def _refusal(reason: str, language: str = "en", query: str = "") -> GroundedAnswer:
    return _fallback_answer(query or "unsupported", language)


def build_nonmanual_reply(query: str) -> GroundedAnswer:
    language, _ = _detect_query_language(query)
    return _fallback_answer(query, language)


def _generate_with_sarvam(query: str,
                          vision: VisionObservation | None,
                          chunks: list[RetrievedChunk],
                          history: list[dict] | None,
                          target_language_code: str,
                          target_language_name: str) -> tuple[dict, dict]:
    usage = {"input_tokens": 0, "output_tokens": 0}
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

    from sarvamai import SarvamAI

    client = SarvamAI(api_subscription_key=settings.sarvam_api_key)
    response = client.chat.completions(
        model=settings.sarvam_model,
        messages=messages,
        temperature=0.2,
        top_p=1,
        max_tokens=2000,
    )

    u = getattr(response, "usage", None)
    if u is not None:
        if hasattr(u, "prompt_tokens"):
            usage["input_tokens"] = int(u.prompt_tokens or 0)
            usage["output_tokens"] = int(getattr(u, "completion_tokens", 0) or 0)
        elif isinstance(u, dict):
            usage["input_tokens"] = int(u.get("prompt_tokens") or 0)
            usage["output_tokens"] = int(u.get("completion_tokens") or 0)

    raw = getattr(response.choices[0].message, "content", "") or ""
    return _parse_answer_json(raw, query, target_language_code, target_language_name), usage


def _generate_with_gemini(query: str,
                          vision: VisionObservation | None,
                          chunks: list[RetrievedChunk],
                          history: list[dict] | None,
                          target_language_code: str,
                          target_language_name: str) -> tuple[dict, dict]:
    usage = {"input_tokens": 0, "output_tokens": 0}

    import google.generativeai as genai

    genai.configure(api_key=settings.gemini_api_key)
    model = genai.GenerativeModel(
        settings.gemini_answer_model,
        system_instruction=SYSTEM_PROMPT,
    )
    response = model.generate_content(
        _build_gemini_prompt(
            query,
            vision,
            chunks,
            history,
            target_language_code,
            target_language_name,
        ),
        generation_config=genai.types.GenerationConfig(
            temperature=0.2,
            top_p=1,
            max_output_tokens=2000,
            response_mime_type="application/json",
        ),
    )

    meta = getattr(response, "usage_metadata", None)
    if meta is not None:
        usage["input_tokens"] = int(getattr(meta, "prompt_token_count", 0) or 0)
        usage["output_tokens"] = int(getattr(meta, "candidates_token_count", 0) or 0)

    raw = getattr(response, "text", "") or ""
    return _parse_answer_json(raw, query, target_language_code, target_language_name), usage


def generate(query: str, vision: VisionObservation | None,
             chunks: list[RetrievedChunk],
             history: list[dict] | None = None,
             answer_model: str = "sarvam") -> tuple[GroundedAnswer, dict]:
    """Call Sarvam 105B for the grounded answer.

    Returns:
        (GroundedAnswer, usage)  where usage = {input_tokens, output_tokens}
    """
    usage = {"input_tokens": 0, "output_tokens": 0}
    target_language_code, target_language_name = _detect_query_language(query)

    if not chunks:
        return _refusal("No manual chunks retrieved.", language=target_language_code, query=query), usage

    try:
        if answer_model == "gemini":
            if not settings.gemini_api_key:
                return GroundedAnswer(
                    answer=(
                        "(Gemini API key not configured. Add GEMINI_API_KEY to api.env "
                        "to enable Gemini answers.)"
                    ),
                    citations=[],
                    confidence="low",
                    manual_supported=False,
                    language=target_language_code,
                ), usage
            parsed, usage = _generate_with_gemini(
                query,
                vision,
                chunks,
                history,
                target_language_code,
                target_language_name,
            )
        else:
            if not settings.sarvam_api_key:
                return GroundedAnswer(
                    answer=(
                        "(Sarvam API key not configured. Add SARVAM_API_KEY to api.env "
                        "to enable multilingual answers.)"
                    ),
                    citations=[],
                    confidence="low",
                    manual_supported=False,
                    language=target_language_code,
                ), usage
            parsed, usage = _generate_with_sarvam(
                query,
                vision,
                chunks,
                history,
                target_language_code,
                target_language_name,
            )
    except Exception as e:
        return GroundedAnswer(
            answer=f"(LLM API error: {e}. Check api.env and try again.)",
            citations=[],
            confidence="low",
            manual_supported=False,
            language=target_language_code,
        ), usage

    citations = [
        Citation(page=int(c.get("page", 0)), chunk_id=str(c.get("chunk_id", "")))
        for c in parsed.get("citations", [])
        if isinstance(c, dict)
    ]
    answer_text = sanitize_answer_text(parsed.get("answer") or raw or "")
    returned_language = parsed.get("language")
    if not _answer_matches_language(
        answer_text,
        returned_language,
        target_language_code,
    ):
        answer_text = _rewrite_answer_in_target_language(
            answer_text,
            target_language_code,
            target_language_name,
        )
        returned_language = target_language_code

    return GroundedAnswer(
        answer=answer_text,
        citations=citations,
        confidence=parsed.get("confidence", "low"),
        manual_supported=bool(parsed.get("manual_supported", False)),
        language=returned_language or target_language_code,
    ), usage
