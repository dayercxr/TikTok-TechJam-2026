# Real ou Fake

Binary classifier (real vs AI-generated) built for robustness under
realistic post-processing: JPEG re-compression, blur, resize/thumbnailing,
sensor noise, color jitter, and center cropping — matching the transform
grid in the problem statement.

## Architecture

- **Backbone**: EfficientNet-B0, ImageNet-pretrained (~4M params, well
  under the 2B cap). Swap in `src/model.py` to try other backbones.
- **Head**: dropout + single linear layer -> sigmoid = P(image is AIGC).
- **Robustness strategy**: the exact transform/parameter grid from the
  spec (`src/transforms.py`) is applied *stochastically during training*
  (not just at test time), so the model learns invariance to these
  corruptions rather than memorizing clean-image artifacts only.

## Project Structure
```bash
.
├── LICENSE
├── README.md
├── requirements.txt
└── src
    ├── dataset.py
    ├── infer.py
    ├── model.py
    ├── robustness.py
    ├── train.py
    └── transforms.py

```

## Setup
```bash
pip install -r requirements.txt
```

## Data layout

```
data/
  train/
    real/   *.jpg
    fake/   *.jpg
  val/
    real/   *.jpg
    fake/   *.jpg
```

Populate from the datasets listed in the problem statement (SID_Set,
CIFAKE, WildFake). Do not train on the WildFake/COCO validation subset
mentioned in the brief — that's reserved as a demo-only benchmark.

## Reproduce results

```bash
cd src
python train.py --data-root ../data --epochs 8 --batch-size 32
python evaluate_robustness.py --data-root ../data --checkpoint ../checkpoints/best.pt
python infer.py --image-dir /path/to/images --checkpoint ../checkpoints/best.pt --out predictions.json
```

- `train.py` fine-tunes the model, freezing the backbone for the first
  epoch (head-only warmup) then unfreezing for full fine-tuning.
- `evaluate_robustness.py` produces `reports/robustness_table.csv`
  (clean vs each transform/parameter accuracy) and
  `reports/error_examples.json` (representative false positives/negatives)
  for the deliverable write-up.
- `infer.py` outputs the required submission format: a JSON list of
  `{"image_path": ..., "pred": ...}`, where `pred` is P(AIGC) in [0,1].

## Limitations / what we'd improve with more time

- Single backbone/architecture compared — an ensemble or a
  frequency-domain branch (e.g. FFT/DCT features) could catch
  generator-specific artifacts that spatial-only CNNs miss.
- Training-time augmentation uses the spec's parameter grid directly,
  which risks over fitting to *these exact* corruption parameters rather
  than corruptions in general; a wider/continuous augmentation range
  would generalize better.
- No calibration step (e.g. temperature scaling) — raw sigmoid outputs
  may not be well-calibrated confidence scores.
- Robustness eval currently re-processes images per-transform serially;
  for larger validation sets this should be batched/parallelized.


## Team contributions
• Gerard Ting Wey Jay
• Tang Shi Rong
• Dayer Cher Xuanrui
• Raphael Ho Zi Jie
• Travis Lim Ee Hng

_Submission for TikTokTechJam 2026_