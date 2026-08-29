"""
Dataset for AIGC vs real image classification.

Expected directory layout (adjust ROOT structure to match whichever of
the datasets in section 5.4 you download):

    data/
      train/
        real/  *.jpg
        fake/  *.jpg
      val/
        real/  *.jpg
        fake/  *.jpg

Label convention: real = 0, fake (AIGC) = 1.
"""

import os
from pathlib import Path

import torch  # type: ignore[import-not-found]
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from transforms import random_robustness_augment

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

IMG_SIZE = 224

VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _list_images(folder: Path):
    if not folder.exists():
        return []
    return [p for p in folder.iterdir() if p.suffix.lower() in VALID_EXTS]


class AIGCDataset(Dataset):
    def __init__(self, root: str, split: str, train_augment: bool = True):
        """
        root: path to the `data/` directory described above
        split: "train" or "val"
        train_augment: if True, apply random robustness augmentation
                        (only meaningful when split == "train")
        """
        self.root = Path(root) / split
        self.train_augment = train_augment and split == "train"

        real_paths = _list_images(self.root / "real")
        fake_paths = _list_images(self.root / "fake")

        if not real_paths and not fake_paths:
            raise FileNotFoundError(
                f"No images found under {self.root}/real or {self.root}/fake. "
                "Check your data directory layout."
            )

        self.samples = [(p, 0) for p in real_paths] + [(p, 1) for p in fake_paths]

        self.to_tensor = T.Compose(
            [
                T.Resize((IMG_SIZE, IMG_SIZE)),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")

        if self.train_augment:
            img = random_robustness_augment(img)

        tensor = self.to_tensor(img)
        return tensor, torch.tensor(label, dtype=torch.float32), str(path)