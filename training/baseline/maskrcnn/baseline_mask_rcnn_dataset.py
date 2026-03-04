import os
import glob
import argparse
from typing import List, Dict

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog

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
# Metrics (your exact formulas) — RAW (no rounding)
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
# Detectron2 Panoptic FPN setup (Mask R-CNN backbone + semantic head)
# -----------------------------
def build_panoptic_predictor(device: str, score_thresh: float, zoo_config: str):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(zoo_config))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(zoo_config)  # auto-download
    if hasattr(cfg.MODEL, "ROI_HEADS"):
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    cfg.MODEL.DEVICE = device
    cfg.freeze()
    predictor = DefaultPredictor(cfg)

    # Find contiguous id for "road" in metadata (stuff classes)
    meta_name = cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "coco_2017_val_panoptic_separated"
    md = MetadataCatalog.get(meta_name)
    stuff_classes = getattr(md, "stuff_classes", None)
    if not stuff_classes:
        raise RuntimeError("Could not retrieve stuff_classes from Detectron2 metadata.")
    try:
        road_contig_id = stuff_classes.index("road")
    except ValueError:
        raise RuntimeError(f"'road' not found in stuff_classes: {stuff_classes}")
    return predictor, road_contig_id

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        "Evaluate Mask R-CNN (Panoptic FPN) using your formulas with one-to-one best match per prediction (ROAD only)."
    )
    parser.add_argument("--images-dir", default="/home/simon/Documents/Master-Thesis/data/baseline_test/test/images/", help="Directory with test images.")
    parser.add_argument("--labels-dir", default="/home/simon/Documents/Master-Thesis/data/baseline_test/test/labels/", help="Directory with YOLO seg .txt files (same basename).")
    parser.add_argument("--config-file", type=str,
                        default="COCO-PanopticSegmentation/panoptic_fpn_R_50_1x.yaml",
                        help="Detectron2 model zoo config (must be Panoptic FPN).")
    parser.add_argument("--road-class-id", type=int, default=4, help="YOLO class index for 'road' in GT labels.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--score-thresh", type=float, default=0.0, help="ROI score threshold (kept permissive).")
    parser.add_argument("--max-images", type=int, default=0, help="Limit images (0 = all).")
    parser.add_argument("--min-area", type=int, default=0, help="Discard predicted components smaller than this (px).")
    parser.add_argument("--match-metric", choices=["iou", "dice", "precision", "recall", "accuracy"], default="iou",
                        help="Primary metric to select the best GT match for each prediction.")
    args = parser.parse_args()

    # Device
    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    print(f"Using device: {device}")

    # Predictor
    predictor, road_contig_id = build_panoptic_predictor(device, args.score_thresh, args.config_file)
    print(f"Model 'road' contiguous id: {road_contig_id}")

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

    # Function used to choose the best match
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
        numpy_image_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # ---- Ground-truth road masks (list) ----
        groundtruth_masks = yolo_segmentation_txt_to_binary_masks(
            txt_path, image_width, image_height, target_class=args.road_class_id
        )

        # ---- Predict panoptic segmentation and extract road stuff masks (list) ----
        with torch.no_grad():
            outputs = predictor(numpy_image_bgr)

        if "panoptic_seg" not in outputs:
            raise RuntimeError(
                "Predictor did not return 'panoptic_seg'. Use a panoptic FPN config like COCO-PanopticSegmentation/panoptic_fpn_R_50_1x.yaml."
            )

        panoptic_seg, segments_info = outputs["panoptic_seg"]
        panoptic_seg = panoptic_seg.to("cpu").numpy()

        predicted_masks = []
        for seg in segments_info:
            if seg.get("isthing", False):
                continue  # keep only stuff
            if seg.get("category_id", -1) != road_contig_id:
                continue
            seg_id = seg["id"]
            m = (panoptic_seg == seg_id).astype(np.uint8)
            if args.min_area > 0 and int(m.sum()) < args.min_area:
                continue
            predicted_masks.append(m)

        # If no GT in this image, skip (mirrors your YOLO script behavior)
        if len(groundtruth_masks) == 0:
            cancelled_images += 1
            continue

        # Build per-prediction scores (one GT per prediction, chosen by primary metric)
        per_prediction_scores = []

        if len(predicted_masks) == 0 and len(groundtruth_masks) > 0:
            # No predictions: add zero-score entries (one per GT) so this image reflects FN-like behavior
            for _ in groundtruth_masks:
                per_prediction_scores.append({"Precision": 0.0, "Recall": 0.0, "Accuracy": 0.0, "Dice": 0.0, "IoU": 0.0})
        else:
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

        # Average per-prediction scores → image-level scores
        image_mean = {
            "Precision": float(np.mean([d["Precision"] for d in per_prediction_scores])) if per_prediction_scores else 0.0,
            "Recall":    float(np.mean([d["Recall"]    for d in per_prediction_scores])) if per_prediction_scores else 0.0,
            "Accuracy":  float(np.mean([d["Accuracy"]  for d in per_prediction_scores])) if per_prediction_scores else 0.0,
            "Dice":      float(np.mean([d["Dice"]      for d in per_prediction_scores])) if per_prediction_scores else 0.0,
            "IoU":       float(np.mean([d["IoU"]       for d in per_prediction_scores])) if per_prediction_scores else 0.0,
        }

        # print(
        #     f"{os.path.basename(image_path)},"
        #     f"{round4(image_mean['Precision'])},"
        #     f"{round4(image_mean['Recall'])},"
        #     f"{round4(image_mean['Accuracy'])},"
        #     f"{round4(image_mean['Dice'])},"
        #     f"{round4(image_mean['IoU'])}"
        # )

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
        print("No images produced valid prediction-level metrics.")

if __name__ == "__main__":
    main()
