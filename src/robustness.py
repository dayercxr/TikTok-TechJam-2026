"""
Robustness Evaluation (deliverable 5.5 #4 and #5).

Runs the trained model on the validation set under:
  - clean (no transform)
  - each transform in the spec, at each parameter value

...and reports accuracy, plus collects representative false
positives/negatives for the error analysis note.

Usage:
    python evaluate_robustness.py --data-root ../data --checkpoint ../checkpoints/best.pt

Outputs:
    ../reports/robustness_table.csv
    ../reports/error_examples.json
"""

import argparse
import csv
import json
import os

import torch  # type: ignore[import-not-found]
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms as T

from dataset import AIGCDataset, IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD
from src.model import AIGCDetector
from src.transforms import TRANSFORM_REGISTRY


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default="../data")
    p.add_argument("--checkpoint", type=str, default="../checkpoints/best.pt")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--report-dir", type=str, default="../reports")
    p.add_argument("--num-error-examples", type=int, default=10)
    return p.parse_args()


def load_model(checkpoint_path, device):
    model = AIGCDetector(pretrained=False).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def to_tensor_fn():
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


@torch.no_grad()
def evaluate_condition(model, dataset, transform_fn, param, device, batch_size, collect_errors, max_errors):
    """
    transform_fn=None -> clean evaluation.
    Otherwise apply transform_fn(img, param) to each PIL image before
    the standard resize/normalize pipeline.
    """
    tensorize = to_tensor_fn()
    correct, total = 0, 0
    errors = []

    loader_indices = range(len(dataset))
    batch_imgs, batch_labels, batch_paths = [], [], []

    def flush():
        nonlocal correct, total
        if not batch_imgs:
            return
        x = torch.stack(batch_imgs).to(device)
        y = torch.tensor(batch_labels, dtype=torch.float32).to(device)
        probs = model.predict_proba(x)
        preds = (probs > 0.5).float()
        correct_mask = (preds == y)
        correct_local = correct_mask.sum().item()
        correct += correct_local
        total += len(batch_labels)

        if collect_errors:
            for i in range(len(batch_labels)):
                if not correct_mask[i] and len(errors) < max_errors:
                    kind = "false_positive" if y[i].item() == 0 else "false_negative"
                    errors.append({
                        "path": batch_paths[i],
                        "true_label": int(y[i].item()),
                        "pred_prob": float(probs[i].item()),
                        "error_type": kind,
                    })

    for idx in loader_indices:
        path, label = dataset.samples[idx]
        img = Image.open(path).convert("RGB")
        if transform_fn is not None:
            img = transform_fn(img, param)
        tensor = tensorize(img)

        batch_imgs.append(tensor)
        batch_labels.append(label)
        batch_paths.append(str(path))

        if len(batch_imgs) == batch_size:
            flush()
            batch_imgs, batch_labels, batch_paths = [], [], []

    flush()  # remainder
    acc = correct / max(total, 1)
    return acc, errors


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.report_dir, exist_ok=True)

    model = load_model(args.checkpoint, device)
    val_ds = AIGCDataset(args.data_root, split="val", train_augment=False)
    print(f"Evaluating on {len(val_ds)} validation images.")

    results = []
    all_errors = []

    # 1) Clean baseline
    acc, errors = evaluate_condition(
        model, val_ds, transform_fn=None, param=None, device=device,
        batch_size=args.batch_size, collect_errors=True, max_errors=args.num_error_examples,
    )
    print(f"[clean] accuracy = {acc:.4f}")
    results.append({"transform": "clean", "param": "-", "accuracy": acc})
    all_errors.extend([{**e, "condition": "clean"} for e in errors])

    # 2) Each transform x each parameter value
    for name, (fn, params) in TRANSFORM_REGISTRY.items():
        for param in params:
            acc, errors = evaluate_condition(
                model, val_ds, transform_fn=fn, param=param, device=device,
                batch_size=args.batch_size, collect_errors=len(all_errors) < args.num_error_examples * 3,
                max_errors=max(0, args.num_error_examples - len(all_errors)),
            )
            print(f"[{name} param={param}] accuracy = {acc:.4f}")
            results.append({"transform": name, "param": str(param), "accuracy": acc})
            all_errors.extend([{**e, "condition": f"{name}={param}"} for e in errors])

    # Write robustness table (deliverable 4)
    csv_path = os.path.join(args.report_dir, "robustness_table.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["transform", "param", "accuracy"])
        writer.writeheader()
        writer.writerows(results)
    print(f"Robustness table written to {csv_path}")

    # Write error examples (deliverable 5 raw material)
    err_path = os.path.join(args.report_dir, "error_examples.json")
    with open(err_path, "w") as f:
        json.dump(all_errors, f, indent=2)
    print(f"Error examples written to {err_path}")


if __name__ == "__main__":
    main()