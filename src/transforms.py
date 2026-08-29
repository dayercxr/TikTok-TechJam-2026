"""
Robustness transforms matching the hackathon spec (section 5.2).

These are implemented as standalone PIL/numpy functions (not just
torchvision built-ins) so that:
  1) they exactly match the parameter grids given in the brief, and
  2) the same functions can be reused for the "clean vs transformed"
     robustness evaluation table required in the deliverables.
"""

import io
import random

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance


# ---- Parameter grids taken directly from section 5.2 ----------------------
JPEG_QUALITIES = [90, 70, 50, 30]
BLUR_SIGMAS = [0.5, 1.0, 2.0]
RESIZE_SCALES = [0.5, 0.25]
NOISE_SIGMAS = [0.02, 0.05, 0.10]          # applied on [0,1] float image
COLOR_JITTER_RANGE = 0.20                   # +/-20% brightness/contrast/sat
CENTER_CROP_FRACTION = 0.80


def jpeg_compress(img: Image.Image, quality: int) -> Image.Image:
    """Simulate social-media re-encoding."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def gaussian_blur(img: Image.Image, sigma: float) -> Image.Image:
    """Out-of-focus simulation."""
    if sigma <= 0:
        return img
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))


def resize_down_up(img: Image.Image, scale: float) -> Image.Image:
    """Thumbnail generation: downscale then upscale back to original size."""
    w, h = img.size
    small = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


def gaussian_noise(img: Image.Image, sigma: float) -> Image.Image:
    """Low-light sensor noise, sigma defined on [0,1] pixel scale."""
    arr = np.asarray(img.convert("RGB")).astype(np.float32) / 255.0
    noise = np.random.normal(0.0, sigma, arr.shape).astype(np.float32)
    noisy = np.clip(arr + noise, 0.0, 1.0)
    return Image.fromarray((noisy * 255).astype(np.uint8))


def color_jitter(img: Image.Image, strength: float = COLOR_JITTER_RANGE) -> Image.Image:
    """Filter apps / auto-enhance simulation: brightness, contrast, saturation."""
    out = img.convert("RGB")
    for enhancer_cls in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
        factor = 1.0 + random.uniform(-strength, strength)
        out = enhancer_cls(out).enhance(factor)
    return out


def center_crop(img: Image.Image, fraction: float = CENTER_CROP_FRACTION) -> Image.Image:
    """Profile-picture style cropping. Returns crop resized back to original size."""
    w, h = img.size
    new_w, new_h = int(w * fraction), int(h * fraction)
    left = (w - new_w) // 2
    top = (h - new_h) // 2
    cropped = img.crop((left, top, left + new_w, top + new_h))
    return cropped.resize((w, h), Image.BILINEAR)


# Named registry: transform name -> (fn, list_of_param_values)
# Used both for randomized training-time augmentation and for the
# fixed-grid robustness evaluation table.
TRANSFORM_REGISTRY = {
    "jpeg": (jpeg_compress, JPEG_QUALITIES),
    "blur": (gaussian_blur, BLUR_SIGMAS),
    "resize": (resize_down_up, RESIZE_SCALES),
    "noise": (gaussian_noise, NOISE_SIGMAS),
    "color_jitter": (color_jitter, [COLOR_JITTER_RANGE]),
    "center_crop": (center_crop, [CENTER_CROP_FRACTION]),
}


def random_robustness_augment(img: Image.Image, p: float = 0.7) -> Image.Image:
    """
    Training-time augmentation: with probability p, apply 1-2 randomly
    chosen transforms (with a randomly chosen parameter from the grid)
    so the model learns to be invariant to realistic post-processing.
    With probability (1-p) the clean image is returned unchanged, so the
    model also keeps strong clean-data accuracy.
    """
    if random.random() > p:
        return img

    names = list(TRANSFORM_REGISTRY.keys())
    n_transforms = random.choice([1, 1, 2])  # mostly single, sometimes stacked
    chosen = random.sample(names, k=n_transforms)

    out = img
    for name in chosen:
        fn, params = TRANSFORM_REGISTRY[name]
        param = random.choice(params)
        try:
            out = fn(out, param)
        except Exception:
            # Robustness of the pipeline itself: skip a transform that
            # fails on a given image rather than crashing training.
            continue
    return out