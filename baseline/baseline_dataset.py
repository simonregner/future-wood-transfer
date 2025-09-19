import os
import glob
import argparse
from typing import List, Dict, Any, Tuple
from collections import defaultdict

import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment

import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog

from pycocotools import mask as maskutil
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def discover_images(images_dir: str) -> List[str]:
    files = []
    for ext in IMG_EXTS:
        files.extend(glob.glob(os.path.join(images_dir, f"*{ext}")))
    files.sort()
    return files


def yolo_seg_txt_to_binary_masks(txt_path: str, img_w: int, img_h: int, target_class: int) -> List[np.ndarray]:
    """
    YOLO segmentation format: class x1 y1 x2 y2 ... xn yn  (normalized to [0,1])
    Returns a list of HxW uint8 masks with values {0,1} for the given class.
    """
    masks = []
    if not os.path.isfile(txt_path):
        return masks

    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            try:
                cls = int(float(parts[0]))
            except ValueError:
                continue
            if cls != target_class:
                continue
            coords = list(map(float, parts[1:]))
            if len(coords) < 6 or len(coords) % 2 != 0:
                continue
            pts = np.array(coords, dtype=np.float32).reshape(-1, 2)
            pts[:, 0] = np.clip(np.round(pts[:, 0] * img_w), 0, img_w - 1)
            pts[:, 1] = np.clip(np.round(pts[:, 1] * img_h), 0, img_h - 1)
            mask = np.zeros((img_h, img_w), dtype=np.uint8)
            cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
            if mask.sum() > 0:
                masks.append(mask)
    return masks


def mask_to_rle(binary_mask: np.ndarray) -> Dict[str, Any]:
    rle = maskutil.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def bbox_from_mask(m: np.ndarray) -> List[float]:
    ys, xs = np.where(m > 0)
    if xs.size == 0:
        return [0.0, 0.0, 0.0, 0.0]
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
    return [float(x1), float(y1), float(x2 - x1 + 1), float(y2 - y1 + 1)]


def iou_masks(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum(dtype=np.float64)
    union = np.logical_or(a, b).sum(dtype=np.float64)
    return float(inter / (union + 1e-9)) if union > 0 else 0.0


def hungarian_iou_match_masks(gt_masks: List[np.ndarray], det_masks: List[np.ndarray]) -> List[Tuple[int, int, float]]:
    if len(gt_masks) == 0 or len(det_masks) == 0:
        return []
    G, D = len(gt_masks), len(det_masks)
    iou_mat = np.zeros((G, D), dtype=np.float32)
    for i in range(G):
        for j in range(D):
            iou_mat[i, j] = iou_masks(gt_masks[i], det_masks[j])
    cost = 1.0 - iou_mat
    rows, cols = linear_sum_assignment(cost)
    return [(int(r), int(c), float(iou_mat[r, c])) for r, c in zip(rows, cols)]


def build_panoptic_cfg(device: str, score_thresh: float, zoo_name: str) -> Tuple[Any, DefaultPredictor]:
    """
    Force a Panoptic FPN model (Mask R-CNN backbone + semantic head) from the model zoo.
    """
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(zoo_name))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(zoo_name)
    # make sure thresholds are permissive for evaluation
    if hasattr(cfg.MODEL, "ROI_HEADS"):
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    if hasattr(cfg.MODEL, "SEM_SEG_HEAD"):
        cfg.MODEL.SEM_SEG_HEAD.SCORE_THRESH_TEST = score_thresh
    cfg.MODEL.DEVICE = device
    cfg.freeze()
    pred = DefaultPredictor(cfg)
    # sanity check: output must contain panoptic_seg
    test_img = np.zeros((32, 32, 3), dtype=np.uint8)
    out = pred(test_img)
    if "panoptic_seg" not in out:
        raise RuntimeError(
            "The predictor did not return panoptic segmentation outputs. "
            "Ensure you are using a *panoptic* config such as "
            "'COCO-PanopticSegmentation/panoptic_fpn_R_50_1x.yaml'."
        )
    return cfg, pred


def main():
    ap = argparse.ArgumentParser("Evaluate Detectron2 Panoptic FPN on YOLO segmentation labels (road only).")
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--labels-dir", required=True)
    ap.add_argument("--road-class-id", type=int, default=4, help="YOLO class id for 'road' in your labels.")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--score-thresh", type=float, default=0.0)
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--config-file", type=str,
                    default="COCO-PanopticSegmentation/panoptic_fpn_R_50_1x.yaml",
                    help="Detectron2 model zoo config (must be Panoptic FPN).")
    args = ap.parse_args()

    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"

    # Build predictor and verify it's panoptic
    cfg, predictor = build_panoptic_cfg(device, args.score_thresh, args.config_file)

    # Find 'road' contiguous id in this model's metadata (stuff classes)
    meta_name = cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else None
    md = MetadataCatalog.get(meta_name) if meta_name else MetadataCatalog.get("__unused")
    stuff_classes = getattr(md, "stuff_classes", None)
    if not stuff_classes:
        # try COCO panoptic default metadata name Detectron2 uses
        md = MetadataCatalog.get("coco_2017_val_panoptic_separated")
        stuff_classes = md.stuff_classes
    try:
        road_contig_id = stuff_classes.index("road")
    except Exception:
        raise RuntimeError(f"'road' not found in stuff_classes: {stuff_classes}")
    print(f"Model 'road' contiguous id: {road_contig_id}")

    # Collect images
    images = discover_images(args.images_dir)
    if args.max_images > 0:
        images = images[: args.max_images]
    if not images:
        raise RuntimeError(f"No images found in {args.images_dir}")

    # COCO containers for single category: road (id=1)
    categories = [{"id": 1, "name": "road"}]
    coco_images, coco_annotations, coco_dets = [], [], []
    ann_id = 1
    matches_iou_sum = 0.0
    gt_count_total = 0

    for img_id, img_path in enumerate(tqdm(images, desc="Evaluating")):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        txt_path = os.path.join(args.labels_dir, stem + ".txt")

        pil_img = Image.open(img_path).convert("RGB")
        W, H = pil_img.size
        img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # ---- Ground truth masks (YOLO seg → road) ----
        gt_masks = yolo_seg_txt_to_binary_masks(txt_path, W, H, target_class=args.road_class_id)

        coco_images.append({"id": img_id, "file_name": os.path.basename(img_path), "width": W, "height": H})
        for m in gt_masks:
            rle = mask_to_rle(m)
            bbox = bbox_from_mask(m)
            coco_annotations.append({
                "id": ann_id, "image_id": img_id, "category_id": 1,
                "segmentation": rle, "area": float(m.sum()), "bbox": bbox, "iscrowd": 0
            })
            ann_id += 1

        # ---- Prediction (Panoptic) ----
        with torch.no_grad():
            outputs = predictor(img_bgr)
        if "panoptic_seg" not in outputs:
            raise RuntimeError(
                "The predictor did not return panoptic segmentation outputs. "
                "Ensure the config is a Panoptic FPN model."
            )

        panoptic_seg, segments_info = outputs["panoptic_seg"]
        panoptic_seg = panoptic_seg.to("cpu").numpy()

        pred_masks = []
        pred_scores = []
        # Panoptic outputs include both thing & stuff. We only keep stuff 'road'.
        for seg in segments_info:
            # Detectron2 panoptic inference attaches 'category_id' as contiguous id
            cat_id = seg["category_id"]
            isthing = seg.get("isthing", False)  # stuff will be False
            if isthing:
                continue
            if cat_id != road_contig_id:
                continue
            seg_id = seg["id"]
            m = (panoptic_seg == seg_id).astype(np.uint8)
            if m.sum() == 0:
                continue
            pred_masks.append(m)
            pred_scores.append(1.0)  # no per-instance score from panoptic head

        # add to COCO dets (segm)
        for m, s in zip(pred_masks, pred_scores):
            rle = mask_to_rle(m)
            bbox = bbox_from_mask(m)
            coco_dets.append({
                "image_id": img_id, "category_id": 1, "segmentation": rle, "score": float(s), "bbox": bbox
            })

        # Mean IoU over dataset (Hungarian), unmatched GT => 0
        matches = hungarian_iou_match_masks(gt_masks, pred_masks)
        matches_iou_sum += sum(m[2] for m in matches)
        gt_count_total += len(gt_masks)

    # ---- COCO segm mAP (road) ----
    coco_gt = COCO()
    coco_gt.dataset = {"images": coco_images, "annotations": coco_annotations, "categories": categories}
    coco_gt.createIndex()

    coco_dt = coco_gt.loadRes(coco_dets) if len(coco_dets) > 0 else COCO()
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="segm")
    coco_eval.params.useCats = 1
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    ap, ap50, ap75 = 0.0, 0.0, 0.0
    if coco_eval.stats is not None:
        ap, ap50, ap75 = map(float, coco_eval.stats[:3])

    dataset_mean_iou = (matches_iou_sum / gt_count_total) if gt_count_total > 0 else 0.0

    print("\n========== RESULTS (ROAD only) ==========")
    print(f"COCO segm mAP@[.5:.95]: {ap:.4f}")
    print(f"COCO segm AP@0.50    : {ap50:.4f}")
    print(f"COCO segm AP@0.75    : {ap75:.4f}")
    print(f"Dataset Mean IoU (Hungarian, unmatched GT=0): {dataset_mean_iou:.4f}")
    print("Done.")


if __name__ == "__main__":
    main()
