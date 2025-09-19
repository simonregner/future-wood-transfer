# text_sam_overlay_cpu.py
# Text-prompted segmentation with auto-checkpoint download (GroundingDINO Swin-B + SAM ViT-H).
# Runs entirely on CPU, shows an overlay, does not save files.

import argparse
from pathlib import Path
import sys
import requests
from tqdm import tqdm
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt

# GroundingDINO
from groundingdino.util.inference import load_model, load_image, predict
# SAM
from segment_anything import sam_model_registry, SamPredictor

CACHE_DIR = Path.home() / ".cache" / "text_sam"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

URLS = {
    "gd_cfg":  "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinB_cfg.py",
    "gd_ckpt": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swinb_cogcoor.pth",
    "sam_ckpt": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
}

PATHS = {
    "gd_cfg":  CACHE_DIR / "GroundingDINO_SwinB_cfg.py",
    "gd_ckpt": CACHE_DIR / "groundingdino_swinb_cogcoor.pth",
    "sam_ckpt": CACHE_DIR / "sam_vit_h_4b8939.pth",
}

def download_file(url: str, dst: Path, desc: str):
    if dst.exists() and dst.stat().st_size > 0:
        return
    print(f"Downloading {desc} …")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dst, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=desc) as pbar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    print(f"Saved to {dst}")

def ensure_checkpoints():
    try:
        download_file(URLS["gd_cfg"], PATHS["gd_cfg"], "GroundingDINO Swin-B config")
        download_file(URLS["gd_ckpt"], PATHS["gd_ckpt"], "GroundingDINO Swin-B checkpoint")
        download_file(URLS["sam_ckpt"], PATHS["sam_ckpt"], "SAM ViT-H checkpoint")
    except Exception as e:
        print("\nERROR while downloading checkpoints:", e)
        sys.exit(1)

def overlay_mask(image_bgr, mask_bool, alpha=0.55):
    color = np.array([0, 255, 0], dtype=np.uint8)
    layer = np.zeros_like(image_bgr, dtype=np.uint8)
    layer[mask_bool] = color
    return cv2.addWeighted(layer, alpha, image_bgr, 1 - alpha, 0.0)

def box_xyxy_to_int(box, W, H):
    x0, y0, x1, y1 = box
    x0 = max(0, min(int(round(x0)), W - 1))
    y0 = max(0, min(int(round(y0)), H - 1))
    x1 = max(0, min(int(round(x1)), W - 1))
    y1 = max(0, min(int(round(y1)), H - 1))
    if x1 <= x0: x1 = min(W - 1, x0 + 1)
    if y1 <= y0: y1 = min(H - 1, y0 + 1)
    return np.array([x0, y0, x1, y1], dtype=np.int32)

def expand_box(box_xyxy, W, H, scale=0.10):
    if scale <= 0:
        return box_xyxy
    x0, y0, x1, y1 = box_xyxy.astype(float)
    w = x1 - x0
    h = y1 - y0
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    x0n = max(0, int(round(cx - w*(1+scale)/2)))
    y0n = max(0, int(round(cy - h*(1+scale)/2)))
    x1n = min(W - 1, int(round(cx + w*(1+scale)/2)))
    y1n = min(H - 1, int(round(cy + h*(1+scale)/2)))
    return np.array([x0n, y0n, x1n, y1n], dtype=np.int32)

def main():
    ap = argparse.ArgumentParser(description="CPU-only text-prompted mask overlay (GroundingDINO SwinB + SAM ViT-H)")
    ap.add_argument("--image", required=True, help="Path to input image")
    ap.add_argument("--text", required=True, help='Text prompt, e.g. "forest road"')
    ap.add_argument("--alpha", type=float, default=0.55, help="Overlay opacity [0..1]")
    ap.add_argument("--box-threshold", type=float, default=0.25, help="GroundingDINO box score threshold")
    ap.add_argument("--text-threshold", type=float, default=0.20, help="GroundingDINO text threshold")
    ap.add_argument("--topk", type=int, default=5, help="Max detections to pass to SAM")
    ap.add_argument("--box-expand", type=float, default=0.15, help="Expand boxes before SAM")
    args = ap.parse_args()

    device = torch.device("cpu")
    print("Using device:", device)

    ensure_checkpoints()

    img_bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img_bgr is None:
        print(f"Could not read image: {args.image}")
        sys.exit(1)
    H, W = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    gd_model = load_model(str(PATHS["gd_cfg"]), str(PATHS["gd_ckpt"]), device=device)
    _, image_gd = load_image(args.image)
    with torch.no_grad():
        boxes_xyxy, logits, phrases = predict(
            model=gd_model,
            image=image_gd,
            caption=args.text,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold
        )

    if boxes_xyxy is None or len(boxes_xyxy) == 0:
        print("No detections for your text prompt at current thresholds.")
        return

    if isinstance(boxes_xyxy, torch.Tensor):
        boxes_xyxy = boxes_xyxy.detach().cpu().numpy()
    else:
        boxes_xyxy = np.asarray(boxes_xyxy)

    if torch.is_tensor(logits):
        scores = logits.detach().sigmoid().cpu().numpy()
    else:
        scores = np.asarray(logits)

    boxes_xyxy = np.ascontiguousarray(boxes_xyxy)
    scores = np.ascontiguousarray(scores)

    order = np.argsort(scores)[::-1][: args.topk]
    order = np.ascontiguousarray(order)
    boxes_xyxy = np.take(boxes_xyxy, order, axis=0)
    scores = np.take(scores, order, axis=0)
    phrases = [phrases[i] for i in order.tolist()]

    sam = sam_model_registry["vit_h"](checkpoint=str(PATHS["sam_ckpt"]))
    sam.to(device)
    predictor = SamPredictor(sam)
    predictor.set_image(img_rgb)

    composite = img_bgr.copy()
    drawn = composite.copy()

    mask_any = False
    for box, sc, phr in zip(boxes_xyxy, scores, phrases):
        bi = box_xyxy_to_int(box, W, H)
        bi = expand_box(bi, W, H, scale=args.box_expand)
        masks, mask_scores, _ = predictor.predict(box=bi.astype(np.float32).copy(), multimask_output=True)
        j = int(np.argmax(mask_scores))
        m = masks[j].astype(bool)

        if m.sum() > 0:
            mask_any = True
            composite = overlay_mask(composite, m, alpha=args.alpha)

        x0, y0, x1, y1 = bi.tolist()
        cv2.rectangle(drawn, (x0, y0), (x1, y1), (0, 255, 0), 2)
        label = f"{phr.strip()} ({sc:.2f})"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(drawn, (x0, max(0, y0 - th - 6)), (x0 + tw + 6, y0), (0, 255, 0), -1)
        cv2.putText(drawn, label, (x0 + 3, y0 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

    if not mask_any:
        print("Warning: SAM did not return any non-empty masks for the text prompt.")

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(composite, cv2.COLOR_BGR2RGB))
    plt.title("Text→Box→Mask (Overlay)")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB))
    plt.title("Detections from text prompt")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
