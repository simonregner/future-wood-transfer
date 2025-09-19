import os
import glob
import argparse
from typing import List, Dict, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from mask2former.config import add_maskformer2_config

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

def parse_id_list(arg: str) -> List[int]:
    items = [s.strip() for s in arg.split(",") if s.strip() != ""]
    return [int(x) for x in items]

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
# Metrics — RAW (no rounding)
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
    return float(2.0 * intersect / (np.sum(pr) + np.sum(gt) + 1e-12))

def iou_raw(gt: np.ndarray, pr: np.ndarray) -> float:
    intersect = np.sum(pr * gt)
    return float(intersect / (np.sum(pr) + np.sum(gt) - intersect + 1e-12))

def round4(x: float) -> float:
    return round(float(x), 4)

# -----------------------------
# Model Wrapper (Detectron2 / Mask2Former)
# -----------------------------
class Mask2FormerWrapper:
    def __init__(self, config_path: str, model_path: str, device: str = "cuda"):
        cfg = get_cfg()
        cfg.set_new_allowed(True)
        add_maskformer2_config(cfg)

        cfg.merge_from_file(config_path)

        cfg.defrost()
        cfg.MODEL.WEIGHTS = model_path
        cfg.MODEL.DEVICE = device
        cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
        cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
        cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = True
        cfg.freeze()

        self.predictor = DefaultPredictor(cfg)

    def predict(self, image_bgr: np.ndarray):
        outputs = self.predictor(image_bgr)
        if "instances" not in outputs:
            return [], []
        instances = outputs["instances"]
        if not instances.has("pred_masks") or not instances.has("pred_classes"):
            return [], []

        pred_masks = instances.pred_masks  # [N,H,W] (bool)
        pred_classes = instances.pred_classes  # [N]
        masks_list, classes_list = [], []
        for mask, cls in zip(pred_masks, pred_classes):
            masks_list.append(mask.to("cpu").numpy().astype("uint8"))
            classes_list.append(int(cls.to("cpu").numpy()))
        return masks_list, classes_list

# -----------------------------
# Core evaluation
# -----------------------------
def evaluate_class_over_dataset(
    yolo_class_id: int,
    image_files: List[str],
    labels_dir: str,
    model: Mask2FormerWrapper,
    mask2yolo_map: Dict[int, int],
    min_area: int,
    match_metric: str,
) -> Tuple[Dict[str, float], int, int]:
    """
    Evaluate a single YOLO class over the dataset using a provided Mask2Former->YOLO mapping.
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

    # Precompute which Mask2Former classes map to this YOLO class
    m2f_classes_for_yolo = {m for m, y in mask2yolo_map.items() if y == yolo_class_id}

    for image_path in tqdm(image_files, desc=f"Class {yolo_class_id}"):
        stem = os.path.splitext(os.path.basename(image_path))[0]
        txt_path = os.path.join(labels_dir, stem + ".txt")

        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            continue
        H, W = image_bgr.shape[:2]

        # GT masks for this YOLO class
        gt_by_class = yolo_segmentation_txt_to_masks_by_class(txt_path, W, H)
        gt_masks = gt_by_class.get(yolo_class_id, [])

        # Predictions once per image
        all_pred_masks, all_pred_classes = model.predict(image_bgr)

        # Collect predicted masks that belong (via mapping) to this YOLO class
        pred_masks = []
        for m, cls_m2f in zip(all_pred_masks, all_pred_classes):
            if cls_m2f in m2f_classes_for_yolo:
                if min_area > 0 and int(m.sum()) < min_area:
                    continue
                pred_masks.append(m)

        if len(gt_masks) == 0:
            cancelled_images += 1
            continue

        per_prediction_scores = []
        if len(pred_masks) == 0 and len(gt_masks) > 0:
            # one zero-entry per GT
            for _ in gt_masks:
                per_prediction_scores.append({"Precision": 0.0, "Recall": 0.0, "Accuracy": 0.0, "Dice": 0.0, "IoU": 0.0})
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
# Main (multi-class with explicit mapping)
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        "Evaluate Mask2Former using YOLO GT with explicit class mapping lists."
    )
    parser.add_argument("--images-dir", default="/home/simon/Documents/Master-Thesis/data/yolo_training_data_road/test/images/", help="Directory with test images.")
    parser.add_argument("--labels-dir", default="/home/simon/Documents/Master-Thesis/data/yolo_training_data_road/test/labels/", help="Directory with YOLO seg .txt files (same basename).")
    parser.add_argument("--config-path", required=True, help="Detectron2 config .yaml file")
    parser.add_argument("--weights-path", required=True, help="Model .pth weights")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])

    # NEW: explicit mapping lists, positionally aligned
    parser.add_argument("--yolo-classes", required=True,
                        help="Comma-separated YOLO class IDs, e.g. '0,1,2'")
    parser.add_argument("--mask-classes", required=True,
                        help="Comma-separated Mask2Former class IDs aligned by position, e.g. '4,7,9'")

    parser.add_argument("--max-images", type=int, default=0, help="Limit images for a quick test (0 = all).")
    parser.add_argument("--min-area", type=int, default=0, help="Discard predicted components smaller than this (pixels).")
    parser.add_argument("--match-metric", choices=["iou", "dice", "precision", "recall", "accuracy"], default="iou",
                        help="Primary metric to select the best GT match for each prediction.")
    args = parser.parse_args()

    # Parse mapping lists
    yolo_ids = parse_id_list(args.yolo_classes)
    mask_ids = parse_id_list(args.mask_classes)

    if len(yolo_ids) != len(mask_ids):
        raise ValueError(f"--yolo-classes and --mask-classes must have the same length "
                         f"(got {len(yolo_ids)} vs {len(mask_ids)}).")

    # Build mapping dicts
    yolo2mask = dict(zip(yolo_ids, mask_ids))
    mask2yolo = {m: y for y, m in yolo2mask.items()}

    print("Class mapping (YOLO -> Mask2Former):")
    for y in yolo_ids:
        print(f"  {y} -> {yolo2mask[y]}")

    # Load model
    model = Mask2FormerWrapper(args.config_path, args.weights_path, args.device)

    # Discover images
    image_files = discover_images(args.images_dir)
    if args.max_images > 0:
        image_files = image_files[: args.max_images]
    if not image_files:
        raise RuntimeError(f"No images found in {args.images_dir}")

    # Evaluate each YOLO class from the provided list
    per_class_results: Dict[int, Dict[str, float]] = {}
    per_class_image_counts: Dict[int, int] = {}
    per_class_cancelled: Dict[int, int] = {}

    for yolo_cls in yolo_ids:
        results, counted, cancelled = evaluate_class_over_dataset(
            yolo_class_id=yolo_cls,
            image_files=image_files,
            labels_dir=args.labels_dir,
            model=model,
            mask2yolo_map=mask2yolo,
            min_area=args.min_area,
            match_metric=args.match_metric,
        )
        per_class_results[yolo_cls] = results
        per_class_image_counts[yolo_cls] = counted
        per_class_cancelled[yolo_cls] = cancelled

    # Report
    print("\n========== PER-CLASS RESULTS ==========")
    for yolo_cls in yolo_ids:
        r = per_class_results[yolo_cls]
        print(f"YOLO Class {yolo_cls} | images used: {per_class_image_counts[yolo_cls]} | cancelled (no GT): {per_class_cancelled[yolo_cls]}")
        print(f"  Precision: {round4(r['Precision'])}")
        print(f"  Recall   : {round4(r['Recall'])}")
        print(f"  Accuracy : {round4(r['Accuracy'])}")
        print(f"  Dice     : {round4(r['Dice'])}")
        print(f"  IoU      : {round4(r['IoU'])}")

    valid_class_means = [per_class_results[cid] for cid in yolo_ids if per_class_image_counts[cid] > 0]
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
