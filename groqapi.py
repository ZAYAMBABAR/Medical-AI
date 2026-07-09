from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx

"""
Single place to configure Groq.

Paste your key here (do NOT commit your real key to GitHub):
"""

import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "#")
GROQ_MODEL = os.getenv("GROQ_MODEL", "")
GROQ_CHAT_MODEL = os.getenv("GROQ_CHAT_MODEL", "")


def groq_is_configured() -> bool:
    return bool(GROQ_API_KEY and "ADD_YOUR_" not in GROQ_API_KEY)


async def groq_chat(
    *,
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: float = 0.4,
) -> Dict[str, Any]:
    """
    Calls Groq OpenAI-compatible Chat Completions endpoint.
    `messages` must be like: [{"role":"system|user|assistant","content":"..."}]
    """
    if not groq_is_configured():
        return {"provider": "local", "reply": "Groq is not configured. Add your key in backend/groqapi.py."}

    url = GROQ_BASE_URL.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": model or GROQ_CHAT_MODEL or GROQ_MODEL,
        "messages": messages,
        "temperature": temperature,
    }

    try:
        async with httpx.AsyncClient(timeout=45.0) as client:
            r = await client.post(url, headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()
    except httpx.HTTPStatusError as e:
        # Common causes: invalid key, rate limits, model not available, bad request.
        msg = f"Groq error: HTTP {e.response.status_code}. Check GROQ_API_KEY / model / quota."
        return {"provider": "groq_error", "reply": msg}
    except httpx.RequestError:
        return {"provider": "groq_error", "reply": "Groq error: network/connection issue."}

    content = (
        data.get("choices", [{}])[0].get("message", {}).get("content")
        or data.get("choices", [{}])[0].get("text")
        or ""
    ).strip()
    if not content:
        return {"provider": "groq_error", "reply": "Groq error: empty response."}
    return {"provider": "groq", "model": payload["model"], "reply": content}


async def groq_tts(text: str, voice: str = "troy") -> bytes:
    """
    Calls Groq Audio Speech endpoint using the official Async SDK.
    """
    if not groq_is_configured():
        raise ValueError("Groq API key not configured")

    from groq import AsyncGroq
    client = AsyncGroq(api_key=GROQ_API_KEY)

    # Orpheus model has a 200 character limit
    safe_text = text[:200]

    try:
        # Using the official Async SDK method
        # Note: Orpheus voices currently include: troy, stella, leo, aura, hazel, stark
        response = await client.audio.speech.create(
            model="canopylabs/orpheus-v1-english",
            voice=voice,
            input=safe_text,
            response_format="wav"
        )
        # Binary response content is usually in .content or .read()
        if hasattr(response, 'content'):
            return response.content
        return await response.read()

    except Exception as e:
        print(f"Groq Async SDK TTS Error: {type(e).__name__} - {str(e)}")
        # Fallback to direct httpx if SDK fails
        url = GROQ_BASE_URL.rstrip("/") + "/audio/speech"
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": "canopylabs/orpheus-v1-english",
            "input": safe_text,
            "voice": voice,
            "response_format": "wav",
        }
        async with httpx.AsyncClient(timeout=60.0) as client_httpx:
            r = await client_httpx.post(url, headers=headers, json=payload)
            if r.status_code != 200:
                err_msg = f"Groq TTS failed: {r.status_code} - {r.text}"
                print(err_msg)
                raise ValueError(err_msg)
            return r.content


