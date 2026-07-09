from typing import Any, Dict, List, Optional

import httpx

try:
    from .groqapi import groq_chat, groq_is_configured, GROQ_MODEL
except ImportError:
    from groqapi import groq_chat, groq_is_configured, GROQ_MODEL

async def generate_medical_report(
    *,
    scan_type: str,
    top_label: str,
    confidence: float,
    scores: List[Dict[str, Any]],
    danger_level: int = 1,
    patient_age: Optional[int] = None,
    patient_sex: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Provider-agnostic report generation:
    - If GROQ_API_KEY is present -> Groq OpenAI-compatible endpoint
    - Else if OPENAI_API_KEY is present -> OpenAI
    - Else -> returns a local templated report (no network)
    """
    prompt = _build_prompt(
        scan_type=scan_type,
        top_label=top_label,
        confidence=confidence,
        scores=scores,
        danger_level=danger_level,
        patient_age=patient_age,
        patient_sex=patient_sex,
    )

    if groq_is_configured():
        data = await groq_chat(
            messages=[
                {"role": "system", "content": "You are a careful medical assistant."},
                {"role": "user", "content": prompt},
            ],
            model=GROQ_MODEL,
            temperature=0.3,
        )
        if data.get("provider") == "groq" and (data.get("reply") or "").strip():
            return {
                "provider": "groq",
                "model": data.get("model", GROQ_MODEL),
                "report": data.get("reply", "").strip(),
            }

    return {"provider": "local", "report": _local_report(scan_type, top_label, confidence, scores, danger_level)}


def _build_prompt(
    *,
    scan_type: str,
    top_label: str,
    confidence: float,
    scores: List[Dict[str, Any]],
    danger_level: int,
    patient_age: Optional[int],
    patient_sex: Optional[str],
) -> str:
    scores_txt = "\n".join(
        [f"- {s['label']}: {float(s['confidence']):.3f}" for s in scores[:5]]
    )
    demographics = []
    if patient_age is not None:
        demographics.append(f"Age: {patient_age}")
    if patient_sex:
        demographics.append(f"Gender: {patient_sex}")

    demo_txt = " / ".join(demographics) if demographics else "Not provided"
    return f"""
You are a careful medical imaging assistant. Based on the model output,
write a VERY BRIEF, simple report. Keep explanations short and non-technical.

Hard safety constraints:
- Do NOT claim a definitive diagnosis.
- Recommend clinical correlation.

Scan type: {scan_type}
Patient: {demo_txt}

Top prediction: {top_label} (confidence {confidence:.3f})
Emergency Triage Level: {danger_level}

Output format (KEEP IT VERY SHORT):
Title: AI Analysis Report
Disease Dangerous Level: {danger_level} (MANDATORY: Use this exact value)
Impression: (1-2 simple sentences)
Next Steps: (1 simple sentence)
Suggested Medicine: (List 1-2 specific common/OTC medicine names relevant to the finding, e.g., 'Amoxicillin if infection is confirmed by doctor' or 'Ibuprofen for inflammation')
WARNING: Please consult a doctor. This list is for educational assistance only.
Treatment: (1 simple sentence)
Nutrition: (Short tip)
Exercise: (Short tip)
Disclaimer: (Standard short disclaimer)
""".strip()


async def _call_openai_compatible(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    provider: str,
) -> Dict[str, Any]:
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a careful medical assistant."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()

    content = (
        data.get("choices", [{}])[0].get("message", {}).get("content")
        or data.get("choices", [{}])[0].get("text")
        or ""
    ).strip()

    return {"provider": provider, "model": model, "report": content}


def _local_report(scan_type: str, top_label: str, confidence: float, scores: List[Dict[str, Any]], danger_level: int) -> str:
    urgency = "Moderate"
    if danger_level >= 3:
        urgency = "High"
    if danger_level <= 1:
        urgency = "Low"

    return (
        f"{scan_type.replace('_', ' ').title()} AI Report\n\n"
        f"Summary:\n"
        f"- The model’s top classification is '{top_label}' with confidence {confidence:.3f}.\n"
        f"- Emergency Triage Level: {danger_level}\n"
        f"- This output should be interpreted alongside clinical context and radiologist review.\n\n"
        f"Impression:\n"
        f"- Top label: {top_label}\n"
        f"- Urgency: {urgency}\n\n"
        f"Suggested next steps:\n"
        f"- Correlate with symptoms, vitals, and history.\n"
        f"- Consider repeat imaging / alternative modality if discordant.\n"
        f"- If concerning features are present, seek specialist/radiology confirmation.\n\n"
        f"Disclaimer:\n"
        f"- This is an AI-generated aid and not a medical diagnosis.\n"
    )

