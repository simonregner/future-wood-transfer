import os
import glob
import argparse
from typing import List, Dict

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

def yolo_segmentation_txt_to_binary_masks(
    txt_path: str, image_width: int, image_height: int, target_class: int
) -> List[np.ndarray]:
    """
    Parse YOLO segmentation .txt (class x1 y1 x2 y2 ...), return binary masks for target_class.
    Coordinates are normalized; we scale to pixels and fill polygons.
    """
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
# Metrics (exact formulas you provided) — RAW (no rounding)
# We'll select the best match by a chosen primary metric, then compute all on that pair.
# -----------------------------
def precision_raw(groundtruth_mask: np.ndarray, predicted_mask: np.ndarray) -> float:
    intersect = np.sum(predicted_mask * groundtruth_mask)
    total_pixel_pred = np.sum(predicted_mask)
    return float(intersect / (total_pixel_pred + 1e-12))

def recall_raw(groundtruth_mask: np.ndarray, predicted_mask: np.ndarray) -> float:
    intersect = np.sum(predicted_mask * groundtruth_mask)
    total_pixel_truth = np.sum(groundtruth_mask)
    return float(intersect / (total_pixel_truth + 1e-12))

def accuracy_raw(groundtruth_mask: np.ndarray, predicted_mask: np.ndarray) -> float:
    intersect = np.sum(predicted_mask * groundtruth_mask)
    union = np.sum(predicted_mask) + np.sum(groundtruth_mask) - intersect
    xor = np.sum(groundtruth_mask == predicted_mask)
    return float(xor / (union + xor - intersect + 1e-12))

def dice_raw(groundtruth_mask: np.ndarray, predicted_mask: np.ndarray) -> float:
    intersect = np.sum(predicted_mask * groundtruth_mask)
    total_sum = np.sum(predicted_mask) + np.sum(groundtruth_mask)
    return float(2.0 * intersect / (total_sum + 1e-12))

def iou_raw(groundtruth_mask: np.ndarray, predicted_mask: np.ndarray) -> float:
    intersect = np.sum(predicted_mask * groundtruth_mask)
    union = np.sum(predicted_mask) + np.sum(groundtruth_mask) - intersect
    return float(intersect / (union + 1e-12))

def round4(x: float) -> float:
    return round(float(x), 4)

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser("Evaluate YOLO seg (ROAD only) using your formulas with one-to-one best-match per prediction.")
    parser.add_argument("--images-dir", default="/home/simon/Documents/Master-Thesis/data/baseline_test/test/images/", help="Directory with test images.")
    parser.add_argument("--labels-dir", default="/home/simon/Documents/Master-Thesis/data/baseline_test/test/labels/", help="Directory with YOLO seg .txt files (same basename).")
    parser.add_argument("--model", required=True, help="Path to YOLO segmentation model (e.g., best.pt or yolo11s-seg.pt).")
    parser.add_argument("--road-class-id", type=int, default=4, help="Class index for 'road' in BOTH GT and predictions.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=0, help="Inference size; 0 keeps model default.")
    parser.add_argument("--max-images", type=int, default=0, help="Limit images for a quick test (0 = all).")
    parser.add_argument("--min-area", type=int, default=0, help="Discard predicted components smaller than this (pixels).")
    parser.add_argument("--match-metric", choices=["iou", "dice", "precision", "recall", "accuracy"], default="iou",
                        help="Primary metric to select the best GT match for each prediction.")
    args = parser.parse_args()

    # Device
    device = 0 if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    print(f"Using device: {'cuda' if device == 0 else 'cpu'}")

    # Load model
    model = YOLO(args.model)

    # Discover images
    image_files = discover_images(args.images_dir)
    if args.max_images > 0:
        image_files = image_files[: args.max_images]
    if not image_files:
        raise RuntimeError(f"No images found in {args.images_dir}")

    # Dataset accumulators
    dataset_sums = {"Precision": 0.0, "Recall": 0.0, "Accuracy": 0.0, "Dice": 0.0, "IoU": 0.0}
    images_counted = 0
    cancelled_images = 0

    print("filename,Precision,Recall,Accuracy,Dice,IoU")

    # Select function used to choose the best match
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

        # Load image
        pil_image = Image.open(image_path).convert("RGB")
        image_width, image_height = pil_image.size
        numpy_image = np.array(pil_image)  # RGB

        # ---- Ground-truth road masks (list) ----
        groundtruth_masks = yolo_segmentation_txt_to_binary_masks(
            txt_path, image_width, image_height, target_class=args.road_class_id
        )

        # ---- YOLO predicted road masks (list) ----
        prediction_kwargs = dict(conf=args.conf, device=device, verbose=False)
        if args.imgsz and args.imgsz > 0:
            prediction_kwargs["imgsz"] = args.imgsz

        results = model.predict(numpy_image, **prediction_kwargs)

        predicted_masks = []
        if results:
            result = results[0]
            if result.masks is not None and result.boxes is not None:
                class_ids = result.boxes.cls.detach().cpu().numpy().astype(int)
                mask_rasters = result.masks.data.detach().cpu().numpy()  # (N, mh, mw) floats in [0,1]
                for idx, class_id in enumerate(class_ids):
                    if class_id != args.road_class_id:
                        continue
                    m = (mask_rasters[idx] >= 0.5).astype(np.uint8)
                    if m.shape[0] != image_height or m.shape[1] != image_width:
                        m = cv2.resize(m, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
                    if args.min_area > 0 and int(m.sum()) < args.min_area:
                        continue
                    predicted_masks.append(m)

        # If no predictions, skip this image
        if len(groundtruth_masks) == 0:
            cancelled_images += 1
            continue

        

        # For each predicted mask: pick ONE GT mask by the chosen match metric, then compute all metrics on that pair.
        per_prediction_scores = []

        if len(predicted_masks) == 0 and len(groundtruth_masks) > 0:
            for gt in groundtruth_masks:
                per_prediction_scores.append({
                    "Precision": 0.0,
                    "Recall":    0.0,
                    "Accuracy":  0.0,
                    "Dice":      0.0,
                    "IoU":       0.0,
                })
            continue

        for predicted_mask in predicted_masks:
            # Choose best GT by primary match metric
            primary_scores = [match_fn(gt, predicted_mask) for gt in groundtruth_masks]
            best_gt_index = int(np.argmax(primary_scores))
            best_gt_mask = groundtruth_masks[best_gt_index]

            # Compute ALL metrics on this SAME pair
            precision_val = precision_raw(best_gt_mask, predicted_mask)
            recall_val    = recall_raw(best_gt_mask, predicted_mask)
            accuracy_val  = accuracy_raw(best_gt_mask, predicted_mask)
            dice_val      = dice_raw(best_gt_mask, predicted_mask)
            iou_val       = iou_raw(best_gt_mask, predicted_mask)

            per_prediction_scores.append({
                "Precision": precision_val,
                "Recall":    recall_val,
                "Accuracy":  accuracy_val,
                "Dice":      dice_val,
                "IoU":       iou_val,
            })

        # Average per-prediction scores -> image-level scores
        image_mean = {
            "Precision": float(np.mean([d["Precision"] for d in per_prediction_scores])),
            "Recall":    float(np.mean([d["Recall"]    for d in per_prediction_scores])),
            "Accuracy":  float(np.mean([d["Accuracy"]  for d in per_prediction_scores])),
            "Dice":      float(np.mean([d["Dice"]      for d in per_prediction_scores])),
            "IoU":       float(np.mean([d["IoU"]       for d in per_prediction_scores])),
        }

        # print(
        #      f"{os.path.basename(image_path)},"
        #      f"{round4(image_mean['Precision'])},"
        #      f"{round4(image_mean['Recall'])},"
        #      f"{round4(image_mean['Accuracy'])},"
        #      f"{round4(image_mean['Dice'])},"
        #      f"{round4(image_mean['IoU'])}"
        #  )

        for k in dataset_sums:
            dataset_sums[k] += image_mean[k]
        images_counted += 1

    # Dataset averages
    if images_counted > 0:

        print(f"Cancelled images (no GT): {cancelled_images} out of {len(image_files)}")

        dataset_mean = {k: dataset_sums[k] / images_counted for k in dataset_sums}
        print("\n========== RESULTS (ROAD only, matched one GT per prediction) ==========")
        print(f"Precision: {round4(dataset_mean['Precision'])}")
        print(f"Recall   : {round4(dataset_mean['Recall'])}")
        print(f"Accuracy : {round4(dataset_mean['Accuracy'])}")
        print(f"Dice     : {round4(dataset_mean['Dice'])}")
        print(f"IoU      : {round4(dataset_mean['IoU'])}")
        print("Done.")
    else:
        print("No images produced valid prediction-level metrics (no predictions found).")

if __name__ == "__main__":
    main()



