#!/usr/bin/env python3
"""
Single-image augmentation preview (images only).

📌 What this script does
- Takes ONE image.
- Applies **every** augmentation once (not random).
- Writes each augmented image to an output folder.
- Also creates a side-by-side **grid preview** image (PNG) so you can see all results at a glance.

🖥️ Usage
    python single_image_augmentation_preview.py \
        --image /path/to/img.jpg \
        --out   /path/to/output_folder \
        [--save-as-grayscale]      # default OFF

Augmentations included:
- original
- darker, brighter
- saltpepper
- jitter
- spring, fall
- flip
- weather effects (if imgaug available), picking the **high‑level augmenters with safe defaults** from the docs: FastSnowyLandscape, Clouds, Fog, Snowflakes, Rain (low‑level *Layer* classes require many parameters and are omitted here to avoid runtime errors).

Dependencies: opencv-python, numpy, (optional) imgaug
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List

# ---------------------
# NumPy 1.24+ compatibility shims for legacy libs (e.g., imgaug)
# ---------------------
# Some older libraries still reference removed aliases like np.bool/np.complex/etc.
# These shims make the script robust without forcing you to downgrade NumPy.
if not hasattr(np, "bool"):
    np.bool = np.bool_           # type: ignore[attr-defined]
if not hasattr(np, "int"):
    np.int = np.int_             # type: ignore[attr-defined]
if not hasattr(np, "float"):
    np.float = np.float_         # type: ignore[attr-defined]
if not hasattr(np, "complex"):
    np.complex = np.complex128   # type: ignore[attr-defined]
if not hasattr(np, "object"):
    np.object = np.object_       # type: ignore[attr-defined]

# Try to import imgaug weather augmenters (optional)
try:
    import imgaug.augmenters as iaa  # type: ignore
    _IMGaug_AVAILABLE = True
except Exception:
    iaa = None  # type: ignore
    _IMGaug_AVAILABLE = False

# ---------------------
# Augmentation helpers
# ---------------------

def adjust_brightness(image: np.ndarray, factor: float) -> np.ndarray:
    return cv2.convertScaleAbs(image, alpha=factor, beta=0)


def add_salt_pepper_noise(image: np.ndarray, amount: float = 0.02, salt_vs_pepper: float = 0.5) -> np.ndarray:
    noisy = image.copy()
    h, w = image.shape[:2]
    num_pixels = h * w
    num_salt = int(amount * num_pixels * salt_vs_pepper)
    num_pepper = int(amount * num_pixels * (1.0 - salt_vs_pepper))

    # Salt (white)
    ys = np.random.randint(0, h, num_salt)
    xs = np.random.randint(0, w, num_salt)
    noisy[ys, xs] = 255

    # Pepper (black)
    ys = np.random.randint(0, h, num_pepper)
    xs = np.random.randint(0, w, num_pepper)
    noisy[ys, xs] = 0

    return noisy


def jitter_colors(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    v_scale = np.random.uniform(0.5, 1.7)
    s_scale = np.random.uniform(0.5, 1.7)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * v_scale, 0, 255)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * s_scale, 0, 255)
    hsv = hsv.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def shift_hue_realistic_dynamic(
    img_bgr: np.ndarray,
    dst_hue_range: Tuple[float, float],
    factor: float,
    sat_scale_range: Tuple[float, float] = (1.0, 1.0),
    val_scale_range: Tuple[float, float] = (1.0, 1.0),
    auto_src: bool = True,
    manual_src_range: Tuple[float, float] = (0.0, 360.0)
) -> np.ndarray:
    factor = np.clip(factor, 0.0, 1.0)
    sat_min, sat_max = max(0.0, sat_scale_range[0]), max(0.0, sat_scale_range[1])
    val_min, val_max = max(0.0, val_scale_range[0]), max(0.0, val_scale_range[1])

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    H, S, V = cv2.split(hsv)

    if auto_src:
        B, G, R = cv2.split(img_bgr.astype(np.float32))
        green_mask = (G > R) & (G > B)
        brown_mask = (R > G) & (R > B)
        mid = (dst_hue_range[0] + dst_hue_range[1]) / 2.0
        src_mask = brown_mask if (90 < mid < 170) else green_mask
        hues = H[src_mask]
        if hues.size > 0:
            lo, hi = np.percentile(hues, [2, 98])
        else:
            lo, hi = manual_src_range[0] / 2.0, manual_src_range[1] / 2.0
        src_min, src_max = lo, hi
    else:
        src_min, src_max = [h / 2.0 for h in manual_src_range]

    dst_min, dst_max = [h / 2.0 for h in dst_hue_range]

    if src_min <= src_max:
        mask = (H >= src_min) & (H <= src_max)
    else:
        mask = (H >= src_min) | (H <= src_max)

    ys, xs = np.where(mask)
    if ys.size == 0:
        return img_bgr

    H_masked = H[ys, xs]
    dst_h_pixels = np.random.uniform(dst_min, dst_max, size=H_masked.shape)
    delta = dst_h_pixels - H_masked
    delta = (delta + 90) % 180 - 90
    H[ys, xs] = (H_masked + factor * delta) % 180

    sat_scales = np.random.uniform(sat_min, sat_max, size=ys.shape)
    val_scales = np.random.uniform(val_min, val_max, size=ys.shape)
    S[ys, xs] = np.clip(S[ys, xs] * ((1 - factor) + factor * sat_scales), 0, 255)
    V[ys, xs] = np.clip(V[ys, xs] * ((1 - factor) + factor * val_scales), 0, 255)

    hsv_shifted = cv2.merge([H, S, V]).astype(np.uint8)
    return cv2.cvtColor(hsv_shifted, cv2.COLOR_HSV2BGR)


def flip_image(image: np.ndarray) -> np.ndarray:
    return cv2.flip(image, 1)

# ---------------------
# Grid preview helper
# ---------------------

def make_grid(images: List[np.ndarray], labels: List[str], tile_w: int = 480) -> np.ndarray:
    resized = []
    for img in images:
        h, w = img.shape[:2]
        scale = tile_w / float(w)
        new = cv2.resize(img, (tile_w, int(round(h * scale))), interpolation=cv2.INTER_LINEAR)
        resized.append(new)

    cols = 2
    rows = (len(resized) + cols - 1) // cols

    cell_h = [0] * rows
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            if idx < len(resized):
                cell_h[r] = max(cell_h[r], resized[idx].shape[0])

    pad = 8
    grid_w = cols * tile_w + (cols + 1) * pad
    grid_h = sum(h for h in cell_h) + (rows + 1) * pad + rows * 32
    canvas = np.full((grid_h, grid_w, 3), 255, dtype=np.uint8)

    y = pad
    font = cv2.FONT_HERSHEY_SIMPLEX
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            if idx >= len(resized):
                break
            img = resized[idx]
            label = labels[idx]
            text_y = y + 24
            cv2.putText(canvas, label, (pad + c * (tile_w + pad), text_y), font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
            iy = y + 32
            ih, iw = img.shape[:2]
            x = pad + c * (tile_w + pad)
            canvas[iy:iy + ih, x:x + iw] = img
        y += cell_h[r] + pad + 32

    return canvas

# ---------------------
# Main
# ---------------------

def main():
    parser = argparse.ArgumentParser(description="Single image augmentation preview (no labels)")
    parser.add_argument("--image", required=True, type=Path, help="Path to input image (.jpg/.png)")
    parser.add_argument("--out", required=True, type=Path, help="Output folder")
    parser.add_argument("--save-as-grayscale", action="store_true", help="Save images as grayscale as well")

    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(args.image))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    base = args.image.stem
    ext = args.image.suffix.lower()

    to_save = [(img, "original")]

    aug_map = {
        "darker": lambda im: adjust_brightness(im, 0.5),
        "brighter": lambda im: adjust_brightness(im, 1.5),
        "saltpepper": lambda im: add_salt_pepper_noise(im, amount=0.02, salt_vs_pepper=0.5),
        "jitter": jitter_colors,
        "spring": lambda im: shift_hue_realistic_dynamic(
            im, dst_hue_range=(100, 160), factor=0.8,
            sat_scale_range=(1.0, 1.3), val_scale_range=(1.0, 1.1), auto_src=True
        ),
        "fall": lambda im: shift_hue_realistic_dynamic(
            im, dst_hue_range=(10, 25), factor=0.9,
            sat_scale_range=(0.7, 1.0), val_scale_range=(0.8, 1.0), auto_src=True
        ),
        "flip": flip_image,
    }

    # Add weather augmenters if imgaug is available
    if _IMGaug_AVAILABLE:
        try:
            import imgaug.augmenters as iaa  # re-import to access weather augmenters
            # As per https://imgaug.readthedocs.io/en/latest/source/api_augmenters_weather.html
            aug_map.update({
            "Clouds": lambda im: iaa.Clouds()(images=[im])[0],
            "Fog": lambda im: iaa.Fog()(images=[im])[0],
            "Snowflakes": lambda im: iaa.Snowflakes(density=0.15)(images=[im])[0],
            "Rain": lambda im: iaa.Rain()(images=[im])[0],
        })
        except Exception as e:
            print(f"[Note] Failed to load imgaug weather augmenters: {e}")

    for name, fn in aug_map.items():
        aug_img = fn(img)
        to_save.append((aug_img, name))

    grid_images, grid_names = [], []
    for im, name in to_save:
        out_img = args.out / f"{base}_{name}{ext}"
        if args.save_as_grayscale:
            im_to_write = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        else:
            im_to_write = im
        cv2.imwrite(str(out_img), im_to_write)
        grid_images.append(im)
        grid_names.append(name)

    grid = make_grid(grid_images, grid_names, tile_w=480)
    grid_path = args.out / f"{base}_ALL_PREVIEW.png"
    cv2.imwrite(str(grid_path), grid)

    if not _IMGaug_AVAILABLE:
        print("[Note] imgaug not installed or failed to import. Weather effects were skipped.")
    print(f"Saved {len(to_save)} images to: {args.out}")
    print(f"Preview grid: {grid_path}")


if __name__ == "__main__":
    main()
