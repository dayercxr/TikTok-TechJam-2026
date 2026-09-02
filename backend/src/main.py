from __future__ import annotations

from contextlib import asynccontextmanager
from io import BytesIO
from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from .config import settings
from .model import Detector, ModelNotReadyError
from .schemas import HealthResponse, ModelMetadata, PredictionResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.detector = Detector(
        checkpoint_path=settings.model_path,
        device=settings.device,
        ai_threshold=settings.ai_threshold,
    )
    yield


app = FastAPI(
    title="AI Image Detector API",
    description="Upload an image and receive a probabilistic authenticity screening result.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _detector(request: Request) -> Detector:
    return request.app.state.detector


@app.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    detector = _detector(request)
    return HealthResponse(
        service="ok",
        model_ready=detector.ready,
        model_path=str(settings.model_path),
        model_error=detector.load_error,
    )


@app.get("/metadata", response_model=ModelMetadata)
async def metadata(request: Request) -> ModelMetadata:
    return ModelMetadata(**_detector(request).metadata())


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: Request, file: UploadFile = File(...)) -> PredictionResponse:
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Please upload an image file.")

    # Read one byte past the limit so oversized uploads can be rejected without
    # buffering an unbounded request body in memory.
    contents = await file.read(settings.max_upload_bytes + 1)
    if len(contents) > settings.max_upload_bytes:
        limit_mb = settings.max_upload_bytes // (1024 * 1024)
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=f"Image must be smaller than {limit_mb} MB.")

    try:
        image = Image.open(BytesIO(contents)).convert("RGB")
    except (UnidentifiedImageError, OSError) as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="The uploaded file is not a readable image.") from error

    try:
        result: dict[str, Any] = _detector(request).predict(image)
    except ModelNotReadyError as error:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)) from error

    return PredictionResponse(filename=file.filename or "upload", **result)
