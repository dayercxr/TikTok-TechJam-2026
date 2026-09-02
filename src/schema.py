from __future__ import annotations

from typing import Dict, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    service: str
    model_ready: bool
    model_path: str
    model_error: Optional[str] = None


class ModelMetadata(BaseModel):
    model_ready: bool
    model_version: Optional[str] = None
    task: Optional[str] = None
    class_names: list[str] = Field(default_factory=list)
    image_size: int
    ai_threshold: float


class PredictionResponse(BaseModel):
    filename: str
    verdict: str
    ai_probability: float = Field(ge=0.0, le=1.0)
    authentic_probability: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    top_class: str
    class_probabilities: Dict[str, float]
    model_version: str
    caveat: str

