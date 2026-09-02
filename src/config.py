from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    model_path: Path
    device: str
    ai_threshold: float
    max_upload_bytes: int
    cors_origins: tuple[str, ...]


def load_settings() -> Settings:
    raw_origins = os.getenv("CORS_ORIGINS", "http://localhost:5173")
    origins = tuple(origin.strip() for origin in raw_origins.split(",") if origin.strip())
    max_upload_mb = max(1, _int_env("MAX_UPLOAD_MB", 10))
    return Settings(
        model_path=_resolve_path(os.getenv("MODEL_PATH", "artifacts/sid_detector.pt")),
        device=os.getenv("DEVICE", "auto"),
        ai_threshold=min(1.0, max(0.0, _float_env("AI_THRESHOLD", 0.50))),
        max_upload_bytes=max_upload_mb * 1024 * 1024,
        cors_origins=origins or ("http://localhost:5173",),
    )


settings = load_settings()

