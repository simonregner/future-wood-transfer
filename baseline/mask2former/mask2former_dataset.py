import os
import glob
import argparse
from typing import List
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo

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

def yolo_segmentation_txt_to_binary_masks(txt_path, image_width, image_height, target_class):
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
# Metrics
# -----------------------------
def precision_raw(gt, pr): return float(np.sum(pr * gt) / (np.sum(pr) + 1e-12))
def recall_raw(gt, pr): return float(np.sum(pr * gt) / (np.sum(gt) + 1e-12))
def accuracy_raw(gt, pr):
    intersect = np.sum(pr * gt)
    union = np.sum(pr) + np.sum(gt) - intersect
    xor = np.sum(gt == pr)
    return float(xor / (union + xor - intersect + 1e-12))
def dice_raw(gt, pr): return float(2 * np.sum(pr * gt) / (np.sum(pr) + np.sum(gt) + 1e-12))
def iou_raw(gt, pr): return float(np.sum(pr * gt) / (np.sum(pr) + np.sum(gt) - np.sum(pr * gt) + 1e-12))
def round4(x): return round(float(x), 4)

# -----------------------------
# Model Wrapper (Detectron2)
# -----------------------------
class Mask2FormerWrapper:
        
    def __init__(self, config_path, model_path, device="cuda"):
        cfg = get_cfg()

        # Allow new keys & register custom cfg options for Mask2Former before merging
        cfg.set_new_allowed(True)
        add_maskformer2_config(cfg)

        # Merge ONLY your Mask2Former YAML (do NOT merge PanopticFPN model_zoo YAMLs)
        cfg.merge_from_file(config_path)

        cfg.defrost()
        cfg.MODEL.WEIGHTS = model_path
        cfg.MODEL.DEVICE = device

        # Optional test switches (only if your YAML supports them)
        # Keep them if you need instance/panoptic/semantic outputs.
        cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
        cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
        cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = True
        # cfg.MODEL.MASK_FORMER.TEST.SCORE_THRESH_TEST = 0.5  # for instance; name varies by repo/version

        cfg.freeze()
        self.predictor = DefaultPredictor(cfg)

    def predict(self, image_bgr):
        outputs = self.predictor(image_bgr)
        instances = outputs["instances"]
        pred_masks = instances.pred_masks  # [N, H, W]
        pred_classes = instances.pred_classes  # [N]

        masks_list, classes_list = [], []
        for mask, cls in zip(pred_masks, pred_classes):
            mask_np = mask.cpu().numpy().astype("uint8")
            masks_list.append(mask_np)
            classes_list.append(int(cls.cpu().numpy()))
        return masks_list, classes_list

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser("Evaluate Detectron2 Mask2Former with YOLO GT.")
    parser.add_argument("--images-dir", default="/home/simon/Documents/Master-Thesis/data/baseline_test/test/images/")
    parser.add_argument("--labels-dir", default="/home/simon/Documents/Master-Thesis/data/baseline_test/test/labels/")
    parser.add_argument("--road-class-id", type=int, default=4, help="YOLO GT class index for 'road'.")
    parser.add_argument("--config-path", required=True, help="Detectron2 config .yaml file")
    parser.add_argument("--weights-path", required=True, help="Model .pth weights")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--max-images", type=int, default=0)
    args = parser.parse_args()

    model = Mask2FormerWrapper(args.config_path, args.weights_path, args.device)

    image_files = discover_images(args.images_dir)
    if args.max_images > 0:
        image_files = image_files[: args.max_images]

    dataset_sums = {"Precision":0.0, "Recall":0.0, "Accuracy":0.0, "Dice":0.0, "IoU":0.0}
    images_counted = 0
    cancelled_images = 0

    for image_path in tqdm(image_files, desc="Evaluating"):
        stem = os.path.splitext(os.path.basename(image_path))[0]
        txt_path = os.path.join(args.labels_dir, stem + ".txt")

        image_bgr = cv2.imread(image_path)
        H, W = image_bgr.shape[:2]

        # GT road masks
        gt_masks = yolo_segmentation_txt_to_binary_masks(txt_path, W, H, args.road_class_id)

        # Model prediction
        pred_masks, pred_classes = model.predict(image_bgr)
        # filter only road-class predictions (you might need to adjust the ID)
        road_pred_masks = [m for m, c in zip(pred_masks, pred_classes) if c == 0]

        road_pred_masks = sorted(road_pred_masks, key=lambda m: np.sum(m), reverse=True)

        #print(f"Image: {stem}, GT roads: {len(gt_masks)}, Predicted roads: {len(road_pred_masks)}")

        if len(gt_masks) == 0:
            cancelled_images += 1
            continue

        per_prediction_scores = []
        if len(road_pred_masks) == 0:
            print("No road predictions for this image.")
            for _ in gt_masks:
                per_prediction_scores.append({"Precision":0.0,"Recall":0.0,"Accuracy":0.0,"Dice":0.0,"IoU":0.0})
        else:
            for i, pr in enumerate(road_pred_masks):
                best_gt = max(gt_masks, key=lambda gt: iou_raw(gt, pr))
                per_prediction_scores.append({
                    "Precision": precision_raw(best_gt, pr),
                    "Recall":    recall_raw(best_gt, pr),
                    "Accuracy":  accuracy_raw(best_gt, pr),
                    "Dice":      dice_raw(best_gt, pr),
                    "IoU":       iou_raw(best_gt, pr),
                })
                if i >= len(gt_masks):
                    #print("Max lengh: ", i)
                    break  # only evaluate as many predictions as there are GT masks

        image_mean = {k: float(np.mean([d[k] for d in per_prediction_scores])) for k in dataset_sums}
        for k in dataset_sums:
            dataset_sums[k] += image_mean[k]
        images_counted += 1

    if images_counted > 0:
        dataset_mean = {k: dataset_sums[k] / images_counted for k in dataset_sums}
        print("\n========== RESULTS ==========")
        for k, v in dataset_mean.items():
            print(f"{k}: {round4(v)}")
        print(f"Cancelled images (no GT): {cancelled_images}")
    else:
        print("No valid images processed.")

if __name__ == "__main__":
    main()
