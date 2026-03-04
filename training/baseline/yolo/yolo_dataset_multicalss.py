import os
import glob
import argparse
from typing import List, Dict, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from ultralytics import YOLO

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

def parse_classes_arg(classes_arg: str) -> List[int]:
    items = [s.strip() for s in classes_arg.split(",") if s.strip() != ""]
    return sorted(list({int(x) for x in items}))

def yolo_segmentation_txt_to_masks_by_class(
    txt_path: str, image_width: int, image_height: int
) -> Dict[int, List[np.ndarray]]:
    """
    Parse a YOLO segmentation .txt and return {class_id: [binary masks]}.
    Each line: class x1 y1 x2 y2 ... xn yn (normalized)
    """
    by_class: Dict[int, List[np.ndarray]] = {}
    if not os.path.isfile(txt_path):
        return by_class

    with open(txt_path, "r") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            try:
                class_id = int(float(parts[0]))
            except ValueError:
                continue

            coords = list(map(float, parts[1:]))
            if len(coords) < 6 or len(coords) % 2 != 0:
                continue

            points = np.array(coords, dtype=np.float32).reshape(-1, 2)
            points[:, 0] = np.clip(np.round(points[:, 0] * image_width),  0, image_width  - 1)
            points[:, 1] = np.clip(np.round(points[:, 1] * image_height), 0, image_height - 1)

            mask = np.zeros((image_height, image_width), dtype=np.uint8)
            cv2.fillPoly(mask, [points.astype(np.int32)], 1)
            if mask.sum() > 0:
                by_class.setdefault(class_id, []).append(mask)
    return by_class

# -----------------------------
# Metrics (your exact formulas) — RAW (no rounding)
# -----------------------------
def precision_raw(groundtruth_mask: np.ndarray, predicted_mask: np.ndarray) -> float:
    intersect = np.sum(predicted_mask * groundtruth_mask)
    return float(intersect / (np.sum(predicted_mask) + 1e-12))

def recall_raw(groundtruth_mask: np.ndarray, predicted_mask: np.ndarray) -> float:
    intersect = np.sum(predicted_mask * groundtruth_mask)
    return float(intersect / (np.sum(groundtruth_mask) + 1e-12))

def accuracy_raw(groundtruth_mask: np.ndarray, predicted_mask: np.ndarray) -> float:
    intersect = np.sum(predicted_mask * groundtruth_mask)
    union = np.sum(predicted_mask) + np.sum(groundtruth_mask) - intersect
    xor = np.sum(groundtruth_mask == predicted_mask)
    return float(xor / (union + xor - intersect + 1e-12))

def dice_raw(groundtruth_mask: np.ndarray, predicted_mask: np.ndarray) -> float:
    intersect = np.sum(predicted_mask * groundtruth_mask)
    return float(2.0 * intersect / (np.sum(predicted_mask) + np.sum(groundtruth_mask) + 1e-12))

def iou_raw(groundtruth_mask: np.ndarray, predicted_mask: np.ndarray) -> float:
    intersect = np.sum(predicted_mask * groundtruth_mask)
    return float(intersect / (np.sum(predicted_mask) + np.sum(groundtruth_mask) - intersect + 1e-12))

def round4(x: float) -> float:
    return round(float(x), 4)

# -----------------------------
# Core evaluation (one GT per prediction, chosen by a primary metric)
# -----------------------------
def evaluate_class_over_dataset(
    class_id: int,
    image_files: List[str],
    labels_dir: str,
    model: YOLO,
    device_for_ultra,  # 0 for cuda, "cpu" for cpu
    imgsz: int,
    conf_thresh: float,
    min_area: int,
    match_metric: str,
) -> Tuple[Dict[str, float], int, int]:
    """
    Returns:
      per_class_dataset_mean (dict of 5 metrics),
      images_counted_for_class,
      cancelled_images_for_class (no GT for that class)
    """
    dataset_sums = {"Precision": 0.0, "Recall": 0.0, "Accuracy": 0.0, "Dice": 0.0, "IoU": 0.0}
    images_counted = 0
    cancelled_images = 0

    match_fn_map = {
        "iou": iou_raw,
        "dice": dice_raw,
        "precision": precision_raw,
        "recall": recall_raw,
        "accuracy": accuracy_raw,
    }
    match_fn = match_fn_map[match_metric]

    # We’ll run YOLO once per image and cache predicted masks for all classes
    for image_path in tqdm(image_files, desc=f"Class {class_id}"):
        stem = os.path.splitext(os.path.basename(image_path))[0]
        txt_path = os.path.join(labels_dir, stem + ".txt")

        # Load image
        pil_image = Image.open(image_path).convert("RGB")
        image_width, image_height = pil_image.size
        numpy_image = np.array(pil_image)  # RGB

        # ---- Ground truth masks by class ----
        gt_by_class = yolo_segmentation_txt_to_masks_by_class(txt_path, image_width, image_height)
        gt_masks = gt_by_class.get(class_id, [])

        # ---- YOLO prediction masks by class (built once per image) ----
        pred_kwargs = dict(conf=conf_thresh, device=device_for_ultra, verbose=False)
        if imgsz and imgsz > 0:
            pred_kwargs["imgsz"] = imgsz

        results_list = model.predict(numpy_image, **pred_kwargs)
        predicted_by_class: Dict[int, List[np.ndarray]] = {}

        if results_list:
            result = results_list[0]
            if result.masks is not None and result.boxes is not None:
                cls_ids = result.boxes.cls.detach().cpu().numpy().astype(int)
                mask_rasters = result.masks.data.detach().cpu().numpy()  # (N, mh, mw) in [0,1]
                for idx, cls in enumerate(cls_ids):
                    mask_bin = (mask_rasters[idx] >= 0.5).astype(np.uint8)
                    if mask_bin.shape[0] != image_height or mask_bin.shape[1] != image_width:
                        mask_bin = cv2.resize(mask_bin, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
                    if min_area > 0 and int(mask_bin.sum()) < min_area:
                        continue
                    predicted_by_class.setdefault(cls, []).append(mask_bin)

        pred_masks = predicted_by_class.get(class_id, [])

        # Skip if no GT for this class in this image (matches your previous behavior)
        if len(gt_masks) == 0:
            cancelled_images += 1
            continue

        per_prediction_scores = []

        # If there are GT masks but no predictions => contribute zeros (FN-like) — one zero per GT
        if len(pred_masks) == 0 and len(gt_masks) > 0:
            for _ in gt_masks:
                per_prediction_scores.append({"Precision": 0.0, "Recall": 0.0, "Accuracy": 0.0, "Dice": 0.0, "IoU": 0.0})
        else:
            # For each prediction, pick ONE GT (by match metric), then compute all metrics on that same pair
            for predicted_mask in pred_masks:
                primary_scores = [match_fn(gt, predicted_mask) for gt in gt_masks]
                best_gt = gt_masks[int(np.argmax(primary_scores))]
                per_prediction_scores.append({
                    "Precision": precision_raw(best_gt, predicted_mask),
                    "Recall":    recall_raw(best_gt, predicted_mask),
                    "Accuracy":  accuracy_raw(best_gt, predicted_mask),
                    "Dice":      dice_raw(best_gt, predicted_mask),
                    "IoU":       iou_raw(best_gt, predicted_mask),
                })

        # Aggregate prediction-level → image-level (mean of predictions)
        if per_prediction_scores:
            image_mean = {
                "Precision": float(np.mean([d["Precision"] for d in per_prediction_scores])),
                "Recall":    float(np.mean([d["Recall"]    for d in per_prediction_scores])),
                "Accuracy":  float(np.mean([d["Accuracy"]  for d in per_prediction_scores])),
                "Dice":      float(np.mean([d["Dice"]      for d in per_prediction_scores])),
                "IoU":       float(np.mean([d["IoU"]       for d in per_prediction_scores])),
            }
            for k in dataset_sums:
                dataset_sums[k] += image_mean[k]
            images_counted += 1

    per_class_mean = {k: (dataset_sums[k] / images_counted) if images_counted > 0 else 0.0 for k in dataset_sums}
    return per_class_mean, images_counted, cancelled_images

# -----------------------------
# Main (multi-class)
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        "Evaluate YOLO segmentation on a LIST of classes, using your formulas with one-to-one best match per prediction."
    )
    parser.add_argument("--images-dir", default="/home/simon/Documents/Master-Thesis/data/yolo_road_test/test/images/", help="Directory with test images.")
    parser.add_argument("--labels-dir", default="/home/simon/Documents/Master-Thesis/data/yolo_road_test/test/labels/", help="Directory with YOLO seg .txt files (same basename).")
    parser.add_argument("--model", required=True, help="Path to YOLO segmentation model (e.g., best.pt or yolo11s-seg.pt).")
    parser.add_argument("--classes", required=True, help="Comma-separated list of class IDs to evaluate, e.g. '0,1,2,4'.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=0, help="Inference size; 0 keeps model default.")
    parser.add_argument("--max-images", type=int, default=0, help="Limit images for a quick test (0 = all).")
    parser.add_argument("--min-area", type=int, default=0, help="Discard predicted components smaller than this (pixels).")
    parser.add_argument("--match-metric", choices=["iou", "dice", "precision", "recall", "accuracy"], default="iou",
                        help="Primary metric to select the best GT match for each prediction.")
    args = parser.parse_args()

    # Device for Ultralytics: 0 == CUDA, "cpu" == CPU
    device_for_ultra = 0 if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    print(f"Using device: {'cuda' if device_for_ultra == 0 else 'cpu'}")

    # Load model
    model = YOLO(args.model)

    # Discover images
    image_files = discover_images(args.images_dir)
    if args.max_images > 0:
        image_files = image_files[: args.max_images]
    if not image_files:
        raise RuntimeError(f"No images found in {args.images_dir}")

    # Parse class list
    class_ids = parse_classes_arg(args.classes)
    print(f"Evaluating classes: {class_ids}")

    # Evaluate class-by-class
    per_class_results: Dict[int, Dict[str, float]] = {}
    per_class_image_counts: Dict[int, int] = {}
    per_class_cancelled: Dict[int, int] = {}

    for cid in class_ids:
        results, counted, cancelled = evaluate_class_over_dataset(
            class_id=cid,
            image_files=image_files,
            labels_dir=args.labels_dir,
            model=model,
            device_for_ultra=device_for_ultra,
            imgsz=args.imgsz,
            conf_thresh=args.conf,
            min_area=args.min_area,
            match_metric=args.match_metric,
        )
        per_class_results[cid] = results
        per_class_image_counts[cid] = counted
        per_class_cancelled[cid] = cancelled

    # Print per-class and macro mean
    print("\n========== PER-CLASS RESULTS ==========")
    for cid in class_ids:
        r = per_class_results[cid]
        print(f"Class {cid} | images used: {per_class_image_counts[cid]} | cancelled (no GT): {per_class_cancelled[cid]}")
        print(f"  Precision: {round4(r['Precision'])}")
        print(f"  Recall   : {round4(r['Recall'])}")
        print(f"  Accuracy : {round4(r['Accuracy'])}")
        print(f"  Dice     : {round4(r['Dice'])}")
        print(f"  IoU      : {round4(r['IoU'])}")

    # Macro-average over the requested classes (only those with images_counted > 0)
    valid_class_means = [per_class_results[cid] for cid in class_ids if per_class_image_counts[cid] > 0]
    if len(valid_class_means) > 0:
        macro_mean = {
            "Precision": float(np.mean([r["Precision"] for r in valid_class_means])),
            "Recall":    float(np.mean([r["Recall"]    for r in valid_class_means])),
            "Accuracy":  float(np.mean([r["Accuracy"]  for r in valid_class_means])),
            "Dice":      float(np.mean([r["Dice"]      for r in valid_class_means])),
            "IoU":       float(np.mean([r["IoU"]       for r in valid_class_means])),
        }
        print("\n========== MACRO MEAN OVER CLASSES ==========")
        print(f"Precision: {round4(macro_mean['Precision'])}")
        print(f"Recall   : {round4(macro_mean['Recall'])}")
        print(f"Accuracy : {round4(macro_mean['Accuracy'])}")
        print(f"Dice     : {round4(macro_mean['Dice'])}")
        print(f"IoU      : {round4(macro_mean['IoU'])}")
    else:
        print("\nNo class had any images with ground-truth; macro mean not computed.")

    print("Done.")

if __name__ == "__main__":
    main()
