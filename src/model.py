"""Model definition and inference helpers for the AI image detector."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy
from PIL import Image, ImageFilter

import torch
from torchvision import models, transforms


IMAGE_SIZE = 224
CLASS_NAMES = ("real", "ai")


def build_model() -> torch.nn.Module:
    """Create the binary classifier used by both training and serving."""
    network = models.resnet18(weights=None)
    network.fc = torch.nn.Linear(network.fc.in_features, len(CLASS_NAMES))
    return network


PREPROCESS = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


@dataclass
class Prediction:
    label: str
    ai_probability: float
    real_probability: float
    confidence: float
    model_mode: str
    model_note: str


class AIDetector:
    """Loads a trained checkpoint when available and retains a safe demo fallback."""

    def __init__(self, checkpoint_path: Path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path
        self.network: torch.nn.Module | None = None
        self.class_names = list(CLASS_NAMES)
        self.mode = "baseline"
        self.note = "Forensics baseline — train a checkpoint for production use"

        if checkpoint_path.exists():
            try:
                network = build_model()
                payload: Any = torch.load(checkpoint_path, map_location=self.device)
                state_dict = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
                if isinstance(payload, dict) and payload.get("classes"):
                    self.class_names = list(payload["classes"])
                network.load_state_dict(state_dict)
                network.to(self.device)
                network.eval()
                self.network = network
                self.mode = "checkpoint"
                self.note = "ResNet-18 classifier"
            except Exception as exc:  # pragma: no cover - startup resilience
                print(f"Could not load checkpoint: {exc}")

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> Prediction:
        image = image.convert("RGB")
        if self.network is not None:
            tensor = PREPROCESS(image).unsqueeze(0).to(self.device)
            probabilities = torch.softmax(self.network(tensor), dim=1)[0].cpu().numpy()
            probabilities_by_class = dict(zip(self.class_names, probabilities))
            real_probability = float(probabilities_by_class.get("real", probabilities[0]))
            ai_probability = float(probabilities_by_class.get("ai", probabilities[1]))
        else:
            ai_probability = self._baseline_probability(image)
            real_probability = 1.0 - ai_probability

        label = "AI-generated" if ai_probability >= 0.5 else "Likely real"
        confidence = max(ai_probability, real_probability)
        return Prediction(
            label=label,
            ai_probability=round(ai_probability, 4),
            real_probability=round(real_probability, 4),
            confidence=round(confidence, 4),
            model_mode=self.mode,
            model_note=self.note,
        )

    @staticmethod
    def _baseline_probability(image: Image.Image) -> float:
        """A lightweight, intentionally conservative fallback for local demos.

        This is not presented as a trained detector. It combines visual statistics
        that can be useful as a smoke-test signal: edge density, residual noise,
        color saturation, and smoothness. A real deployment should use a checkpoint
        trained on a representative and continually refreshed dataset.
        """
        small = image.resize((256, 256))
        rgb = np.asarray(small, dtype=np.float32) / 255.0
        gray = rgb.mean(axis=2)
        blur = np.asarray(small.filter(ImageFilter.GaussianBlur(radius=1)), dtype=np.float32) / 255.0
        residual = np.abs(rgb - blur).mean()
        edge_x = np.abs(np.diff(gray, axis=1)).mean()
        edge_y = np.abs(np.diff(gray, axis=0)).mean()
        edge_density = edge_x + edge_y
        saturation = rgb.max(axis=2).mean() - rgb.min(axis=2).mean()

        # Keep the output close to the uncertain middle: this fallback is only a
        # transparent demo path and should not make strong provenance claims.
        raw = 0.5 + (0.035 - residual) * 3.0 + (saturation - 0.27) * 0.35 - (edge_density - 0.12) * 0.15
        return float(np.clip(raw, 0.18, 0.82))
