from __future__ import annotations

import argparse
import json
import random
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import torch
from datasets import Dataset, load_dataset
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from .labels import label_names_for_task, map_sid_label
from .model import build_classifier, choose_device


@dataclass(frozen=True)
class TrainConfig:
    dataset_name: str
    task: str
    output: Path
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    image_size: int
    num_workers: int
    seed: int
    device: str
    pretrained: bool
    freeze_backbone_epochs: int
    max_train_samples: Optional[int]
    max_val_samples: Optional[int]
    val_fraction: float


class HuggingFaceImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: Dataset, task: str, image_transform: transforms.Compose) -> None:
        self.dataset = dataset
        self.task = task
        self.image_transform = image_transform

    def __len__(self) -> int:
        return len(self.dataset)

    @staticmethod
    def _as_pil(value: Any) -> Image.Image:
        if isinstance(value, Image.Image):
            return value
        if isinstance(value, dict):
            if value.get("bytes") is not None:
                from io import BytesIO

                return Image.open(BytesIO(value["bytes"]))
            if value.get("path"):
                return Image.open(value["path"])
        if isinstance(value, np.ndarray):
            return Image.fromarray(value)
        raise TypeError(f"Unsupported image value: {type(value)!r}")

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        record = self.dataset[index]
        image = self._as_pil(record["image"]).convert("RGB")
        label = map_sid_label(int(record["label"]), self.task)
        return self.image_transform(image), label


def _try_load_split(dataset_name: str, names: Iterable[str]) -> tuple[Optional[Dataset], list[str]]:
    errors: list[str] = []
    for split_name in names:
        try:
            return load_dataset(dataset_name, split=split_name), errors
        except Exception as error:  # Dataset configurations can expose different split names.
            errors.append(f"{split_name}: {error}")
    return None, errors


def load_train_val(dataset_name: str, val_fraction: float, seed: int) -> tuple[Dataset, Dataset]:
    train = load_dataset(dataset_name, split="train")
    validation, errors = _try_load_split(dataset_name, ("val", "validation"))
    if validation is not None:
        return train, validation

    print("No public val/validation split found; creating a holdout from train.")
    try:
        parts = train.train_test_split(
            test_size=val_fraction,
            seed=seed,
            stratify_by_column="label",
        )
    except (ValueError, TypeError):
        parts = train.train_test_split(test_size=val_fraction, seed=seed)
    if errors:
        print("Split lookup details:", " | ".join(errors))
    return parts["train"], parts["test"]


def limit_dataset(dataset: Dataset, maximum: Optional[int], name: str) -> Dataset:
    if maximum is None or maximum <= 0 or maximum >= len(dataset):
        return dataset
    print(f"Using {maximum:,} samples for {name} (of {len(dataset):,}).")
    return dataset.select(range(maximum))


def make_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08)],
                p=0.25,
            ),
            transforms.ToTensor(),
            normalize,
        ]
    )
    resize_size = int(round(image_size * 256 / 224))
    val_transform = transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transform, val_transform


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def class_weights(dataset: Dataset, task: str, num_classes: int) -> torch.Tensor:
    labels = torch.tensor([map_sid_label(int(label), task) for label in dataset["label"]], dtype=torch.long)
    counts = torch.bincount(labels, minlength=num_classes).float()
    weights = counts.sum() / (num_classes * counts.clamp_min(1.0))
    return weights


def metrics_from_confusion(confusion: torch.Tensor) -> dict[str, Any]:
    true_positives = confusion.diag().float()
    actual = confusion.sum(dim=1).float()
    predicted = confusion.sum(dim=0).float()
    precision = true_positives / predicted.clamp_min(1.0)
    recall = true_positives / actual.clamp_min(1.0)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-8)
    total = confusion.sum().item()
    accuracy = float(true_positives.sum().item() / total) if total else 0.0
    return {
        "accuracy": round(accuracy, 5),
        "macro_f1": round(float(f1.mean().item()), 5),
        "per_class_f1": [round(float(value), 5) for value in f1.tolist()],
        "confusion_matrix": confusion.tolist(),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
    use_amp: bool,
    epoch: int,
) -> float:
    model.train()
    running_loss = 0.0
    autocast_context = torch.cuda.amp.autocast if use_amp else nullcontext
    for batch_index, (images, labels) in enumerate(loader, start=1):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(enabled=True) if use_amp else nullcontext():
            logits = model(images)
            loss = criterion(logits, labels)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        running_loss += loss.item() * images.size(0)
        if batch_index % 50 == 0 or batch_index == len(loader):
            print(f"epoch={epoch} batch={batch_index}/{len(loader)} loss={loss.item():.4f}")
    return running_loss / max(1, len(loader.dataset))


@torch.inference_mode()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> tuple[float, dict[str, Any]]:
    model.eval()
    running_loss = 0.0
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        predictions = logits.argmax(dim=1)
        running_loss += loss.item() * images.size(0)
        for actual, predicted in zip(labels.cpu(), predictions.cpu()):
            confusion[int(actual), int(predicted)] += 1
    return running_loss / max(1, len(loader.dataset)), metrics_from_confusion(confusion)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    config: TrainConfig,
    best_metrics: dict[str, Any],
    num_classes: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    payload = {
        "model_state_dict": state_dict,
        "class_names": label_names_for_task(config.task),
        "task": config.task,
        "image_size": config.image_size,
        "num_classes": num_classes,
        "dataset_name": config.dataset_name,
        "model_version": "sid-efficientnet-b0-v1",
        "best_metrics": best_metrics,
    }
    torch.save(payload, path)


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train an AI image detector on a Hugging Face image dataset.")
    parser.add_argument("--dataset", dest="dataset_name", default="saberzl/SID_Set")
    parser.add_argument("--task", choices=("binary", "multiclass"), default="multiclass")
    parser.add_argument("--output", type=Path, default=Path("../artifacts/sid_detector.pt"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=1)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    args = parser.parse_args()
    return TrainConfig(**vars(args))


def main() -> None:
    config = parse_args()
    if config.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    if config.task not in ("binary", "multiclass"):
        raise ValueError("Unsupported task")
    set_seed(config.seed)
    device = choose_device(config.device)
    print(f"Using device: {device}")

    raw_train, raw_val = load_train_val(config.dataset_name, config.val_fraction, config.seed)
    raw_train = limit_dataset(raw_train, config.max_train_samples, "train")
    raw_val = limit_dataset(raw_val, config.max_val_samples, "validation")
    train_transform, val_transform = make_transforms(config.image_size)
    train_dataset = HuggingFaceImageDataset(raw_train, config.task, train_transform)
    val_dataset = HuggingFaceImageDataset(raw_val, config.task, val_transform)
    num_classes = 2 if config.task == "binary" else 3

    loader_kwargs = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": config.num_workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=False, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **loader_kwargs)

    model = build_classifier(num_classes=num_classes, pretrained=config.pretrained).to(device)
    if config.freeze_backbone_epochs > 0:
        for parameter in model.features.parameters():
            parameter.requires_grad = False

    weights = class_weights(raw_train, config.task, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    best_f1 = -1.0
    history: list[dict[str, Any]] = []

    print(f"Train samples: {len(train_dataset):,}; validation samples: {len(val_dataset):,}; task: {config.task}")
    for epoch in range(1, config.epochs + 1):
        if epoch > config.freeze_backbone_epochs:
            for parameter in model.features.parameters():
                parameter.requires_grad = True
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler, use_amp, epoch)
        val_loss, metrics = validate(model, val_loader, criterion, device, num_classes)
        scheduler.step(metrics["macro_f1"])
        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            **metrics,
        }
        history.append(row)
        print(json.dumps(row))
        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            save_checkpoint(config.output, model, config, metrics, num_classes)
            print(f"Saved best checkpoint to {config.output}")

    history_path = config.output.with_suffix(".history.json")
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Training complete. History written to {history_path}")


if __name__ == "__main__":
    main()
