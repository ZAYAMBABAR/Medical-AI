from typing import Any, Dict, List, Optional

import httpx

try:
    from .groqapi import groq_chat, groq_is_configured
except ImportError:
    from groqapi import groq_chat, groq_is_configured

async def chat_completion(
    *,
    messages: List[Dict[str, str]],
    scan_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Stateless chat completion using Groq/OpenAI (OpenAI-compatible).
    Frontend sends the full message history each time.
    """
    system = (
        "You are a helpful medical AI assistant. "
        "Keep your answers VERY BRIEF and simple. Avoid long explanations. "
        "You must be safe: do not diagnose, and always recommend clinician review."
    )

    context_txt = ""
    if scan_context:
        try:
            context_txt = (
                "\n\nContext (latest AI scan output):\n"
                f"- scan_type: {scan_context.get('scan_type')}\n"
                f"- label: {scan_context.get('label')}\n"
                f"- confidence: {scan_context.get('confidence')}\n"
                f"- Emergency Triage Level: {scan_context.get('danger_level')}\n"
            )
        except Exception:
            context_txt = ""

    msgs = [{"role": "system", "content": system + context_txt}]
    # Expect {role: "user"|"assistant", content: "..."}
    for m in messages[-20:]:
        role = m.get("role", "")
        content = m.get("content", "")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            msgs.append({"role": role, "content": content.strip()})

    if not groq_is_configured():
        return {
            "provider": "local",
            "reply": "Chat is not configured. Set GROQ_API_KEY or OPENAI_API_KEY and restart the server.",
        }

    return await groq_chat(messages=msgs)


async def _call_openai_compatible(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    provider: str,
) -> Dict[str, Any]:
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "temperature": 0.4}

    async with httpx.AsyncClient(timeout=45.0) as client:
        r = await client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()

    content = (
        data.get("choices", [{}])[0].get("message", {}).get("content")
        or data.get("choices", [{}])[0].get("text")
        or ""
    ).strip()

    return {"provider": provider, "model": model, "reply": content}

