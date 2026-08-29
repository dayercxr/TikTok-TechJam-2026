"""
Training loop for the AIGC detector.

Usage:
    python train.py --data-root ../data --epochs 8 --batch-size 32

Saves the best checkpoint (by val accuracy) to ../checkpoints/best.pt
"""

import argparse
import os
import time

import torch  # type: ignore[import-not-found]
import torch.nn as nn  # type: ignore[import-not-found]
from torch.utils.data import DataLoader

from dataset import AIGCDataset
from src.model import AIGCDetector, count_parameters


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default="../data")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--checkpoint-dir", type=str, default="../checkpoints")
    p.add_argument("--freeze-backbone-epochs", type=int, default=1,
                    help="Train only the classifier head for this many epochs first.")
    return p.parse_args()


def set_backbone_trainable(model: AIGCDetector, trainable: bool):
    for name, param in model.backbone.named_parameters():
        if "classifier" not in name:
            param.requires_grad = trainable


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    torch.set_grad_enabled(train)
    for images, labels, _ in loader:
        images, labels = images.to(device), labels.to(device)

        if train:
            optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        if train:
            loss.backward()
            optimizer.step()

        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        total_loss += loss.item() * labels.size(0)

    torch.set_grad_enabled(True)
    return total_loss / max(total, 1), correct / max(total, 1)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    train_ds = AIGCDataset(args.data_root, split="train", train_augment=True)
    val_ds = AIGCDataset(args.data_root, split="val", train_augment=False)
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    model = AIGCDetector(pretrained=True).to(device)
    print(f"Model parameters: {count_parameters(model):,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    for epoch in range(args.epochs):
        set_backbone_trainable(model, trainable=epoch >= args.freeze_backbone_epochs)

        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step()
        dt = time.time() - t0

        print(f"Epoch {epoch+1}/{args.epochs} ({dt:.1f}s) | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = os.path.join(args.checkpoint_dir, "best.pt")
            torch.save({"model_state_dict": model.state_dict(), "val_acc": val_acc}, ckpt_path)
            print(f"  -> saved new best checkpoint ({val_acc:.4f}) to {ckpt_path}")

    print(f"Training complete. Best val acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()