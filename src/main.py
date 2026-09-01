"""FastAPI service for AI-image classification."""

import io
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from .model import AIDetector


ROOT = Path(__file__).resolve().parents[1]
detector = AIDetector(ROOT / "checkpoints" / "ai_detector.pt")

app = FastAPI(title="AI Image Detector API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model_mode": detector.mode, "model_note": detector.note}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, object]:
    if file.content_type not in {"image/jpeg", "image/png", "image/webp", "image/gif"}:
        raise HTTPException(status_code=415, detail="Upload a JPG, PNG, WEBP, or GIF image.")

    contents = await file.read()
    if len(contents) > 15 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Images must be smaller than 15 MB.")

    try:
        image = Image.open(io.BytesIO(contents))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="The uploaded file is not a readable image.") from exc

    result = detector.predict(image)
    return {
        "filename": file.filename or "upload",
        "width": image.width,
        "height": image.height,
        "label": result.label,
        "ai_probability": result.ai_probability,
        "real_probability": result.real_probability,
        "confidence": result.confidence,
        "model_mode": result.model_mode,
        "model_note": result.model_note,
    }
