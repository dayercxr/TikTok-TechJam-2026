"""
Lightweight AIGC detector model.

Backbone: EfficientNet-B0 pretrained on ImageNet (~5.3M params), well
under the hackathon's <2B parameter cap, fast to fine-tune on a single
GPU/Colab instance, and a standard strong baseline for image-forensics
style binary classification.

Swap BACKBONE_NAME below for another torchvision model
(e.g. "resnet50", "convnext_tiny") if you want to compare architectures
for the "innovation & problem insight" write-up.
"""

import torch  # type: ignore[import-not-found]
import torch.nn as nn  # type: ignore[import-not-found]

try:
    from torchvision import models
except ImportError:  # pragma: no cover - handled at runtime when torchvision is unavailable
    class _ModelsFallback:
        def __getattr__(self, name):
            raise ImportError(
                "torchvision is required for this model. Install it with: pip install torchvision"
            )

    models = _ModelsFallback()

BACKBONE_NAME = "efficientnet_b0"


class AIGCDetector(nn.Module):
    def __init__(self, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()

        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)

        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, 1),  # single logit: sigmoid -> P(fake)
        )
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x).squeeze(1)  # (B,) raw logits

    @torch.no_grad()
    def predict_proba(self, x):
        logits = self.forward(x)
        return torch.sigmoid(logits)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    m = AIGCDetector()
    n = count_parameters(m)
    print(f"{BACKBONE_NAME}: {n:,} parameters (limit: 2,000,000,000)")