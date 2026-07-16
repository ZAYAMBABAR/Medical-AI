import base64
import io
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from PIL import Image, ImageEnhance, ImageFilter

# TensorFlow is heavy; import once at module load.
import tensorflow as tf

import httpx
from dotenv import load_dotenv

# Load env variables from .env in backend directory
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

async def chat_completion(
    *,
    messages: List[Dict[str, str]],
    scan_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
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

ScanType = Literal["chest_xray", "bone_xray", "brain_mri"]


@dataclass(frozen=True)
class ModelSpec:
    scan_type: ScanType
    model_path: str
    classes_path: str
    input_size: Tuple[int, int]
    normalize: Literal["0_1", "imagenet"]


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(ROOT, "models")
FRONTEND_DIR = os.path.join(ROOT, "frontend")


MODEL_SPECS: Dict[ScanType, ModelSpec] = {
    "chest_xray": ModelSpec(
        scan_type="chest_xray",
        model_path=os.path.join(MODELS_DIR, "chest_model.h5"),
        classes_path=os.path.join(MODELS_DIR, "chest_model_classes.json"),
        input_size=(224, 224),
        # Your training scripts (bone/brain) use ImageDataGenerator(rescale=1./255),
        # so inference should match that.
        normalize="0_1",
    ),
    "bone_xray": ModelSpec(
        scan_type="bone_xray",
        model_path=os.path.join(MODELS_DIR, "bone_model.h5"),
        classes_path=os.path.join(MODELS_DIR, "bone_classes.json"),
        input_size=(224, 224),
        normalize="0_1",
    ),
    "brain_mri": ModelSpec(
        scan_type="brain_mri",
        model_path=os.path.join(MODELS_DIR, "brain_mri_model.h5"),
        classes_path=os.path.join(MODELS_DIR, "brain_mri_classes.json"),
        input_size=(224, 224),
        normalize="0_1",
    ),
}


def _load_class_mapping(path: str) -> Tuple[List[str], Dict[str, int]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"Invalid class mapping JSON: {path}")

    # Mapping is label -> index. We'll invert safely.
    inv: Dict[int, str] = {}
    for label, idx in raw.items():
        if not isinstance(label, str) or not isinstance(idx, int):
            raise ValueError(f"Invalid class mapping entry in {path}: {label} -> {idx}")
        inv[idx] = label

    labels = [inv[i] for i in sorted(inv.keys())]
    return labels, raw


def _preprocess_image(img: Image.Image, spec: ModelSpec) -> np.ndarray:
    # Always work in RGB for consistency (most Keras models expect 3 channels)
    img = img.convert("RGB")
    img = img.resize(spec.input_size, Image.BILINEAR)
    arr = np.asarray(img).astype("float32")

    if spec.normalize == "0_1":
        arr = arr / 255.0
    else:
        # MobileNetV2/EfficientNet family: use tf.keras.applications preprocessing
        arr = tf.keras.applications.imagenet_utils.preprocess_input(arr)

    arr = np.expand_dims(arr, axis=0)
    return arr


def _softmax_if_needed(logits_or_probs: np.ndarray) -> np.ndarray:
    x = np.asarray(logits_or_probs)
    if x.ndim == 2 and x.shape[0] == 1:
        x = x[0]
    if x.ndim != 1:
        x = x.reshape(-1)
    # Heuristic: if values already sum to ~1 and are all in [0,1], keep.
    s = float(np.sum(x))
    if np.all(x >= 0.0) and np.all(x <= 1.0) and 0.98 <= s <= 1.02:
        return x
    e = np.exp(x - np.max(x))
    return e / np.sum(e)


def _img_to_png_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _create_medical_overlay(original: Image.Image) -> Image.Image:
    """
    Creates a medical-style overlay (red/yellow/orange) without Grad-CAM.
    Approach:
    - Enhance contrast slightly (radiology-like)
    - Compute a "detail/saliency" map using edges + local contrast
    - Threshold into three severity bands and colorize
    - Alpha-blend on the original image
    """
    base = original.convert("RGB")
    w, h = base.size

    # Work on grayscale for saliency map.
    gray = base.convert("L")
    gray = ImageEnhance.Contrast(gray).enhance(1.25)
    gray = ImageEnhance.Sharpness(gray).enhance(1.2)

    # Edge magnitude proxy.
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edges = ImageEnhance.Contrast(edges).enhance(2.0)

    # Local contrast proxy: high-pass via blur subtraction.
    blurred = gray.filter(ImageFilter.GaussianBlur(radius=3))
    highpass = ImageChops_subtract(gray, blurred)
    highpass = ImageEnhance.Contrast(highpass).enhance(2.0)

    # Combine maps.
    e = np.asarray(edges).astype("float32") / 255.0
    hp = np.asarray(highpass).astype("float32") / 255.0
    sal = np.clip(0.65 * e + 0.35 * hp, 0.0, 1.0)

    # Smooth to look like a clinical overlay.
    sal_img = Image.fromarray((sal * 255).astype("uint8"), mode="L").filter(
        ImageFilter.GaussianBlur(radius=2)
    )
    sal = np.asarray(sal_img).astype("float32") / 255.0

    # Quantile thresholds adapt to each image.
    t1 = float(np.quantile(sal, 0.85))
    t2 = float(np.quantile(sal, 0.92))
    t3 = float(np.quantile(sal, 0.97))

    # Create RGBA overlay.
    overlay = np.zeros((h, w, 4), dtype=np.uint8)

    # yellow (mild)
    mask1 = sal >= t1
    overlay[mask1] = np.array([255, 215, 0, 70], dtype=np.uint8)

    # orange (moderate)
    mask2 = sal >= t2
    overlay[mask2] = np.array([255, 140, 0, 110], dtype=np.uint8)

    # red (severe)
    mask3 = sal >= t3
    overlay[mask3] = np.array([255, 0, 0, 140], dtype=np.uint8)

    # Add subtle contour-like accent by thickening the severe mask edges.
    severe = Image.fromarray((mask3.astype(np.uint8) * 255), mode="L").filter(
        ImageFilter.MaxFilter(size=5)
    )
    sev = np.asarray(severe) > 0
    overlay[sev] = np.maximum(overlay[sev], np.array([255, 0, 0, 160], dtype=np.uint8))

    overlay_img = Image.fromarray(overlay, mode="RGBA")
    composed = Image.alpha_composite(base.convert("RGBA"), overlay_img).convert("RGB")
    return composed


def _create_roi_highlight(original: Image.Image) -> Image.Image:
    """
    Finds the most 'intense' area and draws a clinical ROI circle/indicator.
    """
    from PIL import ImageDraw

    base = original.convert("RGB")
    w, h = base.size
    gray = base.convert("L")
    
    # Simple saliency proxy
    edges = gray.filter(ImageFilter.FIND_EDGES)
    arr = np.asarray(edges).astype("float32")
    
    # Find center of mass of brightest 2%
    threshold = np.percentile(arr, 98)
    coords = np.argwhere(arr >= threshold)
    
    if len(coords) == 0:
        return base

    cy, cx = np.mean(coords, axis=0)
    
    # Draw circle on transparent layer
    roi_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(roi_layer)
    
    r = min(w, h) // 6
    draw.ellipse([cx-r, cy-r, cx+r, cy+r], outline=(255, 255, 0, 200), width=4)
    
    return Image.alpha_composite(base.convert("RGBA"), roi_layer).convert("RGB")


def ImageChops_subtract(a: Image.Image, b: Image.Image) -> Image.Image:
    # Tiny helper to avoid importing ImageChops; keeps deps minimal.
    arr_a = np.asarray(a).astype(np.int16)
    arr_b = np.asarray(b).astype(np.int16)
    out = np.clip(arr_a - arr_b + 128, 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="L")


class ModelRegistry:
    def __init__(self) -> None:
        self.models: Dict[ScanType, Any] = {}
        self.labels: Dict[ScanType, List[str]] = {}

    def _load_model_compat(self, model_path: str) -> Any:
        """
        Loads legacy .h5 models robustly on TF/Keras 2.16+ (Keras 3).
        Some older exports include layer config keys (e.g. `quantization_config` or `groups`)
        that Keras 3 might reject or handle differently.
        """

        class DenseCompat(tf.keras.layers.Dense):
            @classmethod
            def from_config(cls, config: Dict[str, Any]) -> "DenseCompat":
                config = dict(config)
                config.pop("quantization_config", None)
                return cls(**config)

        class DepthwiseConv2DCompat(tf.keras.layers.DepthwiseConv2D):
            @classmethod
            def from_config(cls, config: Dict[str, Any]) -> "DepthwiseConv2DCompat":
                config = dict(config)
                # Keras 3 DepthwiseConv2D might be pickier about certain keys
                return cls(**config)

        return tf.keras.models.load_model(
            model_path,
            compile=False,
            custom_objects={
                "Dense": DenseCompat,
                "DepthwiseConv2D": DepthwiseConv2DCompat,
            },
        )

    def load_all(self) -> None:
        for scan_type, spec in MODEL_SPECS.items():
            if not os.path.exists(spec.model_path):
                raise FileNotFoundError(f"Missing model: {spec.model_path}")
            if not os.path.exists(spec.classes_path):
                raise FileNotFoundError(f"Missing classes: {spec.classes_path}")

            labels, _ = _load_class_mapping(spec.classes_path)
            model = self._load_model_compat(spec.model_path)

            self.models[scan_type] = model
            self.labels[scan_type] = labels

    def predict(self, scan_type: ScanType, img: Image.Image) -> Dict[str, Any]:
        if scan_type not in self.models:
            raise KeyError(f"Unknown scan_type: {scan_type}")
        spec = MODEL_SPECS[scan_type]
        model = self.models[scan_type]
        labels = self.labels[scan_type]

        x = _preprocess_image(img, spec)
        y = model.predict(x, verbose=0)
        probs = _softmax_if_needed(y)

        if len(probs) != len(labels):
            # Some models may output extra dims; try to flatten.
            probs = probs.reshape(-1)
        if len(probs) != len(labels):
            raise ValueError(
                f"Model output size ({len(probs)}) does not match labels ({len(labels)}) for {scan_type}"
            )

        top_idx = int(np.argmax(probs))
        top_label = labels[top_idx]
        top_conf = float(probs[top_idx])

        all_scores = [
            {"label": labels[i], "confidence": float(probs[i])} for i in range(len(labels))
        ]
        all_scores.sort(key=lambda x: x["confidence"], reverse=True)

        return {
            "scan_type": scan_type,
            "label": top_label,
            "confidence": top_conf,
            "scores": all_scores,
        }


registry = ModelRegistry()

app = FastAPI(title="Medical AI Imaging Analysis", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    registry.load_all()


# Mount the whole frontend directory to support relative paths like /frontend/assets/...
if os.path.isdir(FRONTEND_DIR):
    app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR), name="frontend")

# Also mount assets at the root level for convenience if needed by some styles
if os.path.isdir(os.path.join(FRONTEND_DIR, "assets")):
    app.mount("/assets", StaticFiles(directory=os.path.join(FRONTEND_DIR, "assets")), name="assets")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    fav_path = os.path.join(FRONTEND_DIR, "assets", "favicon.ico")
    if os.path.exists(fav_path):
        return FileResponse(fav_path)
    from fastapi.responses import Response
    return Response(status_code=204)


@app.get("/", response_class=HTMLResponse)
def index() -> Any:
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if not os.path.exists(index_path):
        return HTMLResponse(
            "<h3>Frontend not found</h3><p>Create <code>frontend/index.html</code>.</p>",
            status_code=200,
        )
    return FileResponse(index_path)


@app.post("/predict/{scan_type}")
async def predict(scan_type: ScanType, file: UploadFile = File(...)) -> Any:
    if scan_type not in MODEL_SPECS:
        raise HTTPException(status_code=400, detail="Invalid scan_type")

    content = await file.read()
    try:
        img = Image.open(io.BytesIO(content))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    pred = registry.predict(scan_type, img)

    # Danger level (1-4) heuristic for UI triage display (not a diagnosis).
    # Requested rule:
    # - If prediction is an abnormal disease (fracture/tumor types/pneumonia/TB etc)
    #   and confidence >= 0.90 -> Level 4
    # - If confidence >= 0.80 and < 0.90 -> Level 3
    # - Normal-like -> Level 1
    # - Otherwise -> Level 2
    label_l = str(pred.get("label", "")).strip().lower()
    conf = float(pred.get("confidence", 0.0))

    normal_like = {"normal", "no", "notumor"}
    abnormal_like = {
        "fracture",
        "pneumonia",
        "tuberculosis",
        "infection",
        "glioma",
        "meningioma",
        "pituitary",
        "tumor",
    }

    if label_l in normal_like:
        danger_level = 1
    elif (label_l in abnormal_like) and conf >= 0.90:
        danger_level = 4
    elif (label_l in abnormal_like) and conf >= 0.80:
        danger_level = 3
    else:
        danger_level = 2

    pred["danger_level"] = int(danger_level)

    # Overlay visualization returned as base64 PNG.
    overlay = _create_medical_overlay(img)
    pred["overlay_png_base64"] = _img_to_png_base64(overlay)

    # ROI Highlight circle
    roi = _create_roi_highlight(img)
    pred["roi_png_base64"] = _img_to_png_base64(roi)

    # Echo original as base64 too for easy UI display.
    pred["input_png_base64"] = _img_to_png_base64(img.convert("RGB"))

    # Auto-generate report (Groq if configured, else local template)
    rep = await generate_medical_report(
        scan_type=pred["scan_type"],
        top_label=pred["label"],
        confidence=pred["confidence"],
        scores=pred["scores"],
        danger_level=pred["danger_level"],
    )
    pred["report_provider"] = rep.get("provider", "local")
    pred["report"] = rep.get("report", "")

    return pred


class ReportRequest(BaseModel):
    scan_type: ScanType
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    scores: List[Dict[str, Any]]
    patient_age: Optional[int] = Field(default=None, ge=0, le=130)
    patient_sex: Optional[str] = None


@app.post("/report")
async def report(req: ReportRequest) -> Any:
    data = await generate_medical_report(
        scan_type=req.scan_type,
        top_label=req.label,
        confidence=req.confidence,
        scores=req.scores,
        patient_age=req.patient_age,
        patient_sex=req.patient_sex,
    )
    return data


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    scan_context: Optional[Dict[str, Any]] = None


@app.post("/chat")
async def chat(req: ChatRequest) -> Any:
    payload = {
        "messages": [{"role": m.role, "content": m.content} for m in req.messages],
        "scan_context": req.scan_context,
    }
    data = await chat_completion(**payload)
    return data




