"""Translate user query into 3-5 manual-style search phrases (English).

We use Gemini for this because it's free-tier and fast, and the rewriter's
output never reaches the user — it's only used for retrieval. The user-facing
final answer comes from Sarvam in the user's language.
"""
from __future__ import annotations

import json
import re

from .config import settings


REWRITE_PROMPT = """You are helping retrieve information from a motorcycle service manual.

User asked (any language): {query}
Visual observation (if any): {vision}

Produce 3-5 short SEARCH PHRASES in ENGLISH, written in the formal style of a
motorcycle service manual. Cover the issue, likely root causes, and the affected
components. Output STRICT JSON array of strings, no commentary.

Example output:
["white smoke from exhaust", "engine coolant burning symptom", "head gasket failure indicator"]
"""


def _parse_json_array(text: str) -> list[str]:
    text = text.strip()
    fenced = re.match(r"```(?:json)?\s*(.+?)\s*```", text, re.S)
    if fenced:
        text = fenced.group(1)
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [str(x).strip() for x in data if str(x).strip()]
    except Exception:
        pass
    return []


def rewrite(query: str, vision_text: str = "") -> tuple[list[str], dict]:
    """Returns (rewrites, {input_tokens, output_tokens})."""
    usage = {"input_tokens": 0, "output_tokens": 0}

    if not settings.gemini_api_key:
        combined = (query + " " + vision_text).strip()
        return ([combined] if combined else [query]), usage

    import google.generativeai as genai
    genai.configure(api_key=settings.gemini_api_key)
    model = genai.GenerativeModel(settings.gemini_rewriter_model)

    prompt = REWRITE_PROMPT.format(
        query=query,
        vision=vision_text or "(none)",
    )
    try:
        resp = model.generate_content(prompt)
        meta = getattr(resp, "usage_metadata", None)
        if meta is not None:
            usage["input_tokens"] = int(getattr(meta, "prompt_token_count", 0) or 0)
            usage["output_tokens"] = int(getattr(meta, "candidates_token_count", 0) or 0)

        out = _parse_json_array(resp.text or "")
        if out:
            if query not in out:
                out.append(query)
            return out[:6], usage
    except Exception:
        pass
    return [query], usage
