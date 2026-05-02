"""Vision LLM — extracts structured visual cues from a bike photo using Gemini."""
from __future__ import annotations

import base64
import json
import re
from io import BytesIO
from typing import Optional

from PIL import Image

from .config import settings
from .models import VisionObservation


VISION_PROMPT = """You are a motorcycle technical inspector.
Look at this image of a bike or bike component and describe the visible issue
in technical terms a service-manual would use.

Identify (omit any field you genuinely cannot see):
  - issue_type: smoke / leak / damage / wear / no_visible_issue
  - color: if smoke or fluid, its color
  - origin: where on the bike (exhaust, engine block, fork, chain, ...)
  - intensity: light / moderate / heavy
  - additional_observations: short phrases about other visible cues

Output STRICT JSON only — no commentary, no markdown fences.
Schema:
{
  "issue_type": "...",
  "color": "...",
  "origin": "...",
  "intensity": "...",
  "additional_observations": ["...", "..."]
}
"""


def _decode_image(image_b64: str) -> Image.Image:
    raw = base64.b64decode(image_b64.split(",", 1)[-1])
    return Image.open(BytesIO(raw)).convert("RGB")


def _parse_json(text: str) -> dict:
    text = text.strip()
    # Strip ``` fences if the model added any.
    fenced = re.match(r"```(?:json)?\s*(.+?)\s*```", text, re.S)
    if fenced:
        text = fenced.group(1)
    try:
        return json.loads(text)
    except Exception:
        return {}


def describe_image(image_b64: Optional[str]) -> tuple[Optional[VisionObservation], dict]:
    """Returns (observation, {input_tokens, output_tokens})."""
    usage = {"input_tokens": 0, "output_tokens": 0}
    if not image_b64:
        return None, usage
    if not settings.gemini_api_key:
        return VisionObservation(
            raw="(Gemini key not set — vision disabled. Add GEMINI_API_KEY to api.env.)"
        ), usage

    import google.generativeai as genai
    genai.configure(api_key=settings.gemini_api_key)

    img = _decode_image(image_b64)
    model = genai.GenerativeModel(settings.gemini_vision_model)
    response = model.generate_content([VISION_PROMPT, img])
    raw = (response.text or "").strip()
    parsed = _parse_json(raw)

    meta = getattr(response, "usage_metadata", None)
    if meta is not None:
        usage["input_tokens"] = int(getattr(meta, "prompt_token_count", 0) or 0)
        usage["output_tokens"] = int(getattr(meta, "candidates_token_count", 0) or 0)

    return VisionObservation(
        issue_type=parsed.get("issue_type"),
        color=parsed.get("color"),
        origin=parsed.get("origin"),
        intensity=parsed.get("intensity"),
        additional_observations=parsed.get("additional_observations") or [],
        raw=raw,
    ), usage


def vision_to_text(obs: Optional[VisionObservation]) -> str:
    """Flatten the structured observation into a single English phrase
    that can be appended to the user's query for retrieval."""
    if not obs:
        return ""
    parts: list[str] = []
    if obs.issue_type:
        parts.append(obs.issue_type)
    if obs.color:
        parts.append(f"{obs.color} color")
    if obs.origin:
        parts.append(f"from {obs.origin}")
    if obs.intensity:
        parts.append(f"{obs.intensity} intensity")
    parts.extend(obs.additional_observations or [])
    return ", ".join(p for p in parts if p)
