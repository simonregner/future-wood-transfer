import os
import glob
import argparse
from typing import List, Dict

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

# -----------------------------
# CITYSCAPES labels (panoptic_id → name)
# -----------------------------
CITYSCAPES_LABELS = {
    0: 'road', 1: 'ego vehicle', 2: 'rectification border', 3: 'out of roi', 4: 'static',
    5: 'dynamic', 6: 'ground', 7: 'road1', 8: 'sidewalk', 9: 'parking', 10: 'rail track',
    11: 'building', 12: 'wall', 13: 'fence', 14: 'guard rail', 15: 'bridge', 16: 'tunnel',
    17: 'pole', 18: 'polegroup', 19: 'traffic light', 20: 'traffic sign', 21: 'vegetation',
    22: 'terrain', 23: 'sky', 24: 'person', 25: 'rider', 26: 'car', 27: 'truck',
    28: 'bus', 29: 'caravan', 30: 'trailer', 31: 'train', 32: 'motorcycle', 33: 'bicycle', -1: 'void'
}

# -----------------------------
# Helpers
# -----------------------------
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

def discover_images(images_dir: str) -> List[str]:
    files = []
    for ext in IMAGE_EXTENSIONS:
        files.extend(glob.glob(os.path.join(images_dir, f"*{ext}")))
    files.sort()
    return files

def yolo_segmentation_txt_to_binary_masks(
    txt_path: str, image_width: int, image_height: int, target_class: int
) -> List[np.ndarray]:
    """Parse YOLO seg .txt and return binary masks for target_class."""
    masks = []
    if not os.path.isfile(txt_path):
        return masks
    with open(txt_path, "r") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            try:
                class_id = int(float(parts[0]))
            except ValueError:
                continue
            if class_id != target_class:
                continue
            coords = list(map(float, parts[1:]))
            if len(coords) < 6 or len(coords) % 2 != 0:
                continue
            points = np.array(coords, dtype=np.float32).reshape(-1, 2)
            points[:, 0] = np.clip(np.round(points[:, 0] * image_width), 0, image_width - 1)
            points[:, 1] = np.clip(np.round(points[:, 1] * image_height), 0, image_height - 1)
            mask = np.zeros((image_height, image_width), dtype=np.uint8)
            cv2.fillPoly(mask, [points.astype(np.int32)], 1)
            if mask.sum() > 0:
                masks.append(mask)
    return masks

# -----------------------------
# Metrics (your exact formulas) — RAW
# -----------------------------
def precision_raw(gt: np.ndarray, pr: np.ndarray) -> float:
    intersect = np.sum(pr * gt)
    return float(intersect / (np.sum(pr) + 1e-12))

def recall_raw(gt: np.ndarray, pr: np.ndarray) -> float:
    intersect = np.sum(pr * gt)
    return float(intersect / (np.sum(gt) + 1e-12))

def accuracy_raw(gt: np.ndarray, pr: np.ndarray) -> float:
    intersect = np.sum(pr * gt)
    union = np.sum(pr) + np.sum(gt) - intersect
    xor = np.sum(gt == pr)
    return float(xor / (union + xor - intersect + 1e-12))

def dice_raw(gt: np.ndarray, pr: np.ndarray) -> float:
    intersect = np.sum(pr * gt)
    return float(2 * intersect / (np.sum(pr) + np.sum(gt) + 1e-12))

def iou_raw(gt: np.ndarray, pr: np.ndarray) -> float:
    intersect = np.sum(pr * gt)
    return float(intersect / (np.sum(pr) + np.sum(gt) - intersect + 1e-12))

def round4(x: float) -> float:
    return round(float(x), 4)

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser("Evaluate Mask2Former with YOLO GT using Medium-style metrics (one-to-one matching).")
    parser.add_argument("--images-dir", default="/home/simon/Documents/Master-Thesis/data/baseline_test/test/images/", help="Test images.")
    parser.add_argument("--labels-dir", default="/home/simon/Documents/Master-Thesis/data/baseline_test/test/labels/", help="YOLO seg GT labels.")
    parser.add_argument("--road-class-id", type=int, default=4, help="YOLO class index for 'road'.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--min-area", type=int, default=0)
    parser.add_argument("--match-metric", choices=["iou", "dice", "precision", "recall", "accuracy"], default="iou")
    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    print(f"Using device: {device}")

    # Load Mask2Former
    processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-small-cityscapes-panoptic")
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        "facebook/mask2former-swin-small-cityscapes-panoptic", use_safetensors=True
    ).to(device)
    model.eval()

    # Get "road" ID from CITYSCAPES_LABELS
    road_ids = [k for k, v in CITYSCAPES_LABELS.items() if v == "road"]
    if not road_ids:
        raise RuntimeError("No 'road' label found in CITYSCAPES_LABELS")
    road_label_id = road_ids[0]

    # Images
    image_files = discover_images(args.images_dir)
    if args.max_images > 0:
        image_files = image_files[: args.max_images]
    if not image_files:
        raise RuntimeError(f"No images found in {args.images_dir}")

    dataset_sums = {"Precision":0.0, "Recall":0.0, "Accuracy":0.0, "Dice":0.0, "IoU":0.0}
    images_counted = 0
    cancelled_images = 0

    print("filename,Precision,Recall,Accuracy,Dice,IoU")

    match_fn_map = {
        "iou": iou_raw,
        "dice": dice_raw,
        "precision": precision_raw,
        "recall": recall_raw,
        "accuracy": accuracy_raw,
    }
    match_fn = match_fn_map[args.match_metric]

    for image_path in tqdm(image_files, desc="Evaluating"):
        stem = os.path.splitext(os.path.basename(image_path))[0]
        txt_path = os.path.join(args.labels_dir, stem + ".txt")

        # GT road masks
        pil_img = Image.open(image_path).convert("RGB")
        W, H = pil_img.size
        gt_masks = yolo_segmentation_txt_to_binary_masks(txt_path, W, H, args.road_class_id)

        # Mask2Former prediction
        inputs = processor(images=pil_img, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        pred_seg = processor.post_process_semantic_segmentation(outputs, target_sizes=[(H, W)])[0].cpu().numpy()

        pred_masks = []
        for seg_id in np.unique(pred_seg):
            if seg_id == road_label_id:
                m = (pred_seg == seg_id).astype(np.uint8)
                if args.min_area > 0 and int(m.sum()) < args.min_area:
                    continue
                pred_masks.append(m)

        # Skip if no GT
        if len(gt_masks) == 0:
            cancelled_images += 1
            continue

        per_prediction_scores = []

        if len(pred_masks) == 0 and len(gt_masks) > 0:
            # FN-like
            for _ in gt_masks:
                per_prediction_scores.append({"Precision":0.0,"Recall":0.0,"Accuracy":0.0,"Dice":0.0,"IoU":0.0})
        else:
            for pr in pred_masks:
                primary_scores = [match_fn(gt, pr) for gt in gt_masks]
                best_gt = gt_masks[int(np.argmax(primary_scores))]

                per_prediction_scores.append({
                    "Precision": precision_raw(best_gt, pr),
                    "Recall":    recall_raw(best_gt, pr),
                    "Accuracy":  accuracy_raw(best_gt, pr),
                    "Dice":      dice_raw(best_gt, pr),
                    "IoU":       iou_raw(best_gt, pr),
                })

        image_mean = {k: float(np.mean([d[k] for d in per_prediction_scores])) if per_prediction_scores else 0.0
                      for k in dataset_sums}

        # print(f"{os.path.basename(image_path)},{round3(image_mean['Precision'])},{round3(image_mean['Recall'])},"
        #       f"{round3(image_mean['Accuracy'])},{round3(image_mean['Dice'])},{round3(image_mean['IoU'])}")

        for k in dataset_sums: dataset_sums[k] += image_mean[k]
        images_counted += 1

    if images_counted > 0:
        print(f"Cancelled images (no GT): {cancelled_images} out of {len(image_files)}")
        dataset_mean = {k: dataset_sums[k] / images_counted for k in dataset_sums}
        print("\n========== RESULTS (ROAD only, Mask2Former, matched one GT per prediction) ==========")
        print(f"Precision: {round4(dataset_mean['Precision'])}")
        print(f"Recall   : {round4(dataset_mean['Recall'])}")
        print(f"Accuracy : {round4(dataset_mean['Accuracy'])}")
        print(f"Dice     : {round4(dataset_mean['Dice'])}")
        print(f"IoU      : {round4(dataset_mean['IoU'])}")
    else:
        print("No valid images processed.")

if __name__ == "__main__":
    main()
