"""
Inference script matching the required deliverable format (5.5 #2):

  "A script that takes an image directory as input and outputs a
   confidence score for each image, indicating the likelihood that
   it is AIGC-generated. The output should be a JSON file containing
   image_path and pred for each image."

Usage:
    python infer.py --image-dir /path/to/images --checkpoint ../checkpoints/best.pt --out predictions.json

Output format (predictions.json):
    [
      {"image_path": "img001.jpg", "pred": 0.9231},
      {"image_path": "img002.jpg", "pred": 0.0142},
      ...
    ]
`pred` is P(image is AIGC-generated), in [0, 1].
"""

import argparse
import json
from pathlib import Path

import torch  # type: ignore[import-not-found]
from PIL import Image
from torchvision import transforms as T

from dataset import IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD
from src.model import AIGCDetector

VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image-dir", type=str, required=True)
    p.add_argument("--checkpoint", type=str, default="../checkpoints/best.pt")
    p.add_argument("--out", type=str, default="predictions.json")
    p.add_argument("--batch-size", type=int, default=32)
    return p.parse_args()


def load_model(checkpoint_path, device):
    model = AIGCDetector(pretrained=False).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(args.checkpoint, device)
    tensorize = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    image_dir = Path(args.image_dir)
    paths = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in VALID_EXTS])
    if not paths:
        raise FileNotFoundError(f"No images found in {image_dir}")

    results = []
    batch_tensors, batch_paths = [], []

    def flush():
        if not batch_tensors:
            return
        x = torch.stack(batch_tensors).to(device)
        with torch.no_grad():
            probs = model.predict_proba(x).cpu().tolist()
        for path, prob in zip(batch_paths, probs):
            results.append({"image_path": str(path), "pred": round(float(prob), 4)})

    for path in paths:
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"Skipping {path}: {e}")
            continue
        batch_tensors.append(tensorize(img))
        batch_paths.append(path)

        if len(batch_tensors) == args.batch_size:
            flush()
            batch_tensors, batch_paths = [], []

    flush()

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Wrote {len(results)} predictions to {args.out}")


if __name__ == "__main__":
    main()