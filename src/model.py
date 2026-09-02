from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from .labels import aggregate_ai_probability, label_names_for_task


class ModelNotReadyError(RuntimeError):
    """Raised when the API has no usable trained checkpoint."""


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_classifier(num_classes: int, pretrained: bool = False) -> torch.nn.Module:
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    network = efficientnet_b0(weights=weights)
    input_features = network.classifier[1].in_features
    network.classifier[1] = torch.nn.Linear(input_features, num_classes)
    return network


def _load_checkpoint(path: Path) -> dict[str, Any]:
    # weights_only was added after the versions supported by this starter.
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


class Detector:
    def __init__(self, checkpoint_path: Path, device: str = "auto", ai_threshold: float = 0.5) -> None:
        self.checkpoint_path = checkpoint_path
        self.device = choose_device(device)
        self.ai_threshold = ai_threshold
        self.model: Optional[torch.nn.Module] = None
        self.model_version = "sid-efficientnet-b0-v1"
        self.task = "multiclass"
        self.class_names = label_names_for_task(self.task)
        self.image_size = 224
        self.load_error: Optional[str] = None
        self.transform = self._make_transform()

        if checkpoint_path.exists():
            try:
                self._load()
            except Exception as error:  # Keep health endpoint available with a useful error.
                self.load_error = f"Could not load checkpoint: {error}"
        else:
            self.load_error = f"Checkpoint not found at {checkpoint_path}"

    def _make_transform(self) -> transforms.Compose:
        resize_size = int(round(self.image_size * 256 / 224))
        return transforms.Compose(
            [
                transforms.Resize(resize_size),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    def _load(self) -> None:
        payload = _load_checkpoint(self.checkpoint_path)
        self.task = payload.get("task", "multiclass")
        self.class_names = payload.get("class_names") or label_names_for_task(self.task)
        self.image_size = int(payload.get("image_size", 224))
        self.model_version = payload.get("model_version", self.model_version)

        state_dict = payload.get("model_state_dict", payload)
        self.model = build_classifier(len(self.class_names), pretrained=False)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        self.transform = self._make_transform()
        self.load_error = None

    @property
    def ready(self) -> bool:
        return self.model is not None and self.load_error is None

    def metadata(self) -> dict[str, Any]:
        return {
            "model_ready": self.ready,
            "model_version": self.model_version if self.ready else None,
            "task": self.task if self.ready else None,
            "class_names": self.class_names if self.ready else [],
            "image_size": self.image_size,
            "ai_threshold": self.ai_threshold,
        }

    def predict(self, image: Image.Image) -> dict[str, Any]:
        if not self.ready:
            raise ModelNotReadyError(self.load_error or "No trained model is loaded")

        assert self.model is not None
        rgb_image = image.convert("RGB")
        tensor = self.transform(rgb_image).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0].detach().cpu().tolist()

        class_probabilities = {
            name: round(float(probability), 4)
            for name, probability in zip(self.class_names, probabilities)
        }
        ai_probability = min(1.0, max(0.0, aggregate_ai_probability(probabilities, self.class_names)))
        authentic_probability = 1.0 - ai_probability
        top_index = max(range(len(probabilities)), key=probabilities.__getitem__)
        top_class = self.class_names[top_index]
        confidence = float(probabilities[top_index])

        return {
            "verdict": (
                "likely_ai_generated_or_manipulated"
                if ai_probability >= self.ai_threshold
                else "likely_authentic"
            ),
            "ai_probability": round(ai_probability, 4),
            "authentic_probability": round(authentic_probability, 4),
            "confidence": round(confidence, 4),
            "top_class": top_class,
            "class_probabilities": class_probabilities,
            "model_version": self.model_version,
            "caveat": "This is a probabilistic screening result, not proof of provenance.",
        }
