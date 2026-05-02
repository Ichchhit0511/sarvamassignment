"""Whapi (whapi.cloud) WhatsApp integration.

Inbound  — webhook receives JSON, we extract text and any image, run the
           pipeline, and send the answer back via Whapi REST.
Outbound — POST to /messages/text with Bearer token.

Whapi docs: https://whapi.readme.io/reference
"""
from __future__ import annotations

import base64
from typing import Optional

import httpx

from .config import settings


def _auth_headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {settings.whapi_token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def send_text(to: str, body: str) -> dict:
    """Send a WhatsApp text message via Whapi."""
    if not settings.whapi_token:
        return {"ok": False, "error": "WHAPI_TOKEN missing in api.env"}
    url = f"{settings.whapi_base_url.rstrip('/')}/messages/text"
    payload = {"to": to, "body": body}
    with httpx.Client(timeout=30.0) as client:
        r = client.post(url, json=payload, headers=_auth_headers())
        try:
            r.raise_for_status()
        except Exception as e:
            return {"ok": False, "error": str(e), "response": r.text}
        return {"ok": True, "response": r.json()}


def fetch_media_b64(media_id: str) -> Optional[str]:
    """Download a media object from Whapi and return as base64 data-URI."""
    if not settings.whapi_token or not media_id:
        return None
    url = f"{settings.whapi_base_url.rstrip('/')}/media/{media_id}"
    with httpx.Client(timeout=30.0) as client:
        r = client.get(url, headers={
            "Authorization": f"Bearer {settings.whapi_token}",
            "Accept": "*/*",
        })
        if r.status_code != 200:
            return None
        ctype = r.headers.get("content-type", "image/jpeg")
        return f"data:{ctype};base64,{base64.b64encode(r.content).decode()}"


def parse_inbound(payload: dict) -> dict:
    """Extract (sender, text, image_b64) from a Whapi webhook payload.

    Whapi delivers messages under either `messages` (list) or directly. We try
    a few shapes defensively.
    """
    msgs = payload.get("messages") or []
    if not msgs and isinstance(payload.get("message"), dict):
        msgs = [payload["message"]]

    if not msgs:
        return {"sender": None, "text": None, "image_b64": None}

    msg = msgs[0]
    sender = msg.get("from") or msg.get("chat_id") or msg.get("sender", {}).get("id")

    text = None
    if isinstance(msg.get("text"), dict):
        text = msg["text"].get("body")
    elif isinstance(msg.get("text"), str):
        text = msg["text"]
    if not text and isinstance(msg.get("caption"), str):
        text = msg["caption"]

    image_b64 = None
    image = msg.get("image") or {}
    media_id = image.get("id") or image.get("media_id")
    if media_id:
        image_b64 = fetch_media_b64(media_id)

    return {"sender": sender, "text": text, "image_b64": image_b64}
