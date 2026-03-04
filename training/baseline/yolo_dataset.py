import os
import glob
import argparse
from typing import List, Dict, Any, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment  # Hungarian

import torch
from ultralytics import YOLO

from pycocotools import mask as maskutil
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# -----------------------------
# Helpers
# -----------------------------
def discover_images(images_dir: str) -> List[str]:
    files = []
    for ext in IMG_EXTS:
        files.extend(glob.glob(os.path.join(images_dir, f"*{ext}")))
    files.sort()
    return files


def yolo_seg_txt_to_binary_masks(
    txt_path: str, img_w: int, img_h: int, target_class: int
) -> List[np.ndarray]:
    """
    Parse a YOLO segmentation .txt file.
    Each line: class x1 y1 x2 y2 ... xn yn  (normalized coords in [0,1])
    Returns a list of HxW uint8 masks (values 0/1) for the given target_class.
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
            # scale to absolute pixel coords
            pts[:, 0] = pts[:, 0] * img_w
            pts[:, 1] = pts[:, 1] * img_h
            pts = np.round(pts).astype(np.int32)

            # clamp inside image
            pts[:, 0] = np.clip(pts[:, 0], 0, img_w - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, img_h - 1)

            mask = np.zeros((img_h, img_w), dtype=np.uint8)
            cv2.fillPoly(mask, [pts], 1)
            if mask.sum() > 0:
                masks.append(mask)
    return masks


def mask_to_rle(binary_mask: np.ndarray) -> Dict[str, Any]:
    """
    Convert HxW {0,1} mask to COCO RLE dict.
    pycocotools expects Fortran order.
    """
    rle = maskutil.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def bbox_from_mask(binary_mask: np.ndarray) -> List[float]:
    ys, xs = np.where(binary_mask > 0)
    if xs.size == 0 or ys.size == 0:
        return [0.0, 0.0, 0.0, 0.0]
    x1, y1 = xs.min(), ys.min()
    x2, y2 = xs.max(), ys.max()
    return [float(x1), float(y1), float(x2 - x1 + 1), float(y2 - y1 + 1)]


def iou_masks(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum(dtype=np.float64)
    union = np.logical_or(a, b).sum(dtype=np.float64)
    if union == 0:
        return 0.0
    return float(inter / (union + 1e-9))


def hungarian_iou_match_masks(gt_masks: List[np.ndarray], det_masks: List[np.ndarray]) -> List[Tuple[int, int, float]]:
    """
    Hungarian matching to maximize total mask IoU between GT and detections.
    Returns (gt_idx, det_idx, iou) for all assigned pairs (no IoU threshold).
    """
    if len(gt_masks) == 0 or len(det_masks) == 0:
        return []
    G, D = len(gt_masks), len(det_masks)
    iou_mat = np.zeros((G, D), dtype=np.float32)
    for i in range(G):
        for j in range(D):
            iou_mat[i, j] = iou_masks(gt_masks[i], det_masks[j])

    cost = 1.0 - iou_mat
    row_ind, col_ind = linear_sum_assignment(cost)
    return [(int(r), int(c), float(iou_mat[r, c])) for r, c in zip(row_ind, col_ind)]


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser("Evaluate YOLO-seg model on YOLO GT — ROAD class only.")
    ap.add_argument("--images-dir", required=True, help="Directory with test images.")
    ap.add_argument("--labels-dir", required=True, help="Directory with YOLO seg .txt files (same basename).")
    ap.add_argument("--model", required=True, help="Path to YOLO segmentation model (e.g., best.pt).")
    ap.add_argument("--road-class-id", type=int, default=0, help="Class index used for 'road' in both GT and predictions.")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for YOLO predictions.")
    ap.add_argument("--imgsz", type=int, default=0, help="Optional inference size; 0 keeps model default.")
    ap.add_argument("--max-images", type=int, default=0, help="Limit images for a quick test (0 = all).")
    ap.add_argument("--min-area", type=int, default=0, help="Discard predicted components with fewer pixels than this.")
    args = ap.parse_args()

    # Device
    device = 0 if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    print(f"Using device: {'cuda' if device == 0 else 'cpu'}")

    # Load YOLO model
    model = YOLO(args.model)

    # Collect images
    image_files = discover_images(args.images_dir)
    if args.max_images > 0:
        image_files = image_files[: args.max_images]
    if not image_files:
        raise RuntimeError(f"No images found in {args.images_dir}")

    # COCO containers (single category: road)
    categories = [{"id": 1, "name": "road"}]  # we map your road class to category_id=1
    coco_images = []
    coco_annotations = []
    coco_dets = []

    ann_id = 1
    matches_iou_sum = 0.0
    gt_count_total = 0

    for img_id, img_path in enumerate(tqdm(image_files, desc="Evaluating")):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        txt_path = os.path.join(args.labels_dir, stem + ".txt")

        # Load image
        pil_img = Image.open(img_path).convert("RGB")
        W, H = pil_img.size
        np_img = np.array(pil_img)

        # --------- Ground truth (YOLO segmentation → masks for road class) ----------
        gt_masks = yolo_seg_txt_to_binary_masks(txt_path, W, H, target_class=args.road_class_id)

        coco_images.append({"id": img_id, "file_name": os.path.basename(img_path), "width": W, "height": H})

        # add GT annotations
        for m in gt_masks:
            rle = mask_to_rle(m)
            bbox = bbox_from_mask(m)
            area = float(m.sum())
            coco_annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,           # road
                "segmentation": rle,
                "area": area,
                "bbox": bbox,
                "iscrowd": 0
            })
            ann_id += 1

        # --------- YOLO Predictions (keep only class == road_class_id) ----------
        pred_kwargs = dict(conf=args.conf, device=device, verbose=False)
        if args.imgsz and args.imgsz > 0:
            pred_kwargs["imgsz"] = args.imgsz

        results = model.predict(np_img, **pred_kwargs)  # model expects RGB (np already RGB from PIL)
        if not results:
            pred_masks = []
            pred_scores = []
        else:
            r = results[0]

            pred_masks = []
            pred_scores = []

            if r.masks is not None and r.boxes is not None:
                # classes and scores align with masks
                cls_ids = r.boxes.cls.detach().cpu().numpy().astype(int) if r.boxes.cls is not None else np.array([], dtype=int)
                confs = r.boxes.conf.detach().cpu().numpy().astype(float) if r.boxes.conf is not None else np.ones_like(cls_ids, dtype=float)

                # Option A: use mask raster (may be smaller than image; we resize)
                masks = r.masks.data.detach().cpu().numpy()  # (N, mh, mw), float in [0,1]
                for i, cls_id in enumerate(cls_ids):
                    if cls_id != args.road_class_id:
                        continue
                    m = (masks[i] >= 0.5).astype(np.uint8)
                    if m.shape[0] != H or m.shape[1] != W:
                        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
                    if args.min_area > 0 and int(m.sum()) < args.min_area:
                        continue
                    pred_masks.append(m)
                    pred_scores.append(float(confs[i]))

                # ----
                # If you prefer polygon filling at original resolution (often crisper):
                # polys = r.masks.xy  # list of arrays in image coords
                # pred_masks = []
                # pred_scores = []
                # for i, cls_id in enumerate(cls_ids):
                #     if cls_id != args.road_class_id:
                #         continue
                #     m = np.zeros((H, W), dtype=np.uint8)
                #     parts = polys[i] if isinstance(polys[i], (list, tuple)) else [polys[i]]
                #     for pts in parts:
                #         if pts is None or len(pts) == 0:
                #             continue
                #         cv2.fillPoly(m, [pts.astype(np.int32)], 1)
                #     if args.min_area > 0 and int(m.sum()) < args.min_area:
                #         continue
                #     pred_masks.append(m)
                #     pred_scores.append(float(confs[i]))

        # --------- Add detections to COCO dets (segm RLE) ----------
        for m, s in zip(pred_masks, pred_scores):
            rle = mask_to_rle(m)
            bbox = bbox_from_mask(m)
            coco_dets.append({
                "image_id": img_id,
                "category_id": 1,
                "segmentation": rle,
                "score": float(s),
                "bbox": bbox
            })

        # --------- Dataset Mean IoU (instance-level, unmatched GT = 0) ----------
        matches = hungarian_iou_match_masks(gt_masks, pred_masks)
        sum_iou = sum(m[2] for m in matches)
        matches_iou_sum += sum_iou
        gt_count_total += len(gt_masks)

    # ---------------- COCO mask AP (segm) ----------------
    coco_gt = COCO()
    coco_gt.dataset = {"images": coco_images, "annotations": coco_annotations, "categories": categories}
    coco_gt.createIndex()

    coco_dt = coco_gt.loadRes(coco_dets) if len(coco_dets) > 0 else COCO()
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="segm")
    coco_eval.params.useCats = 1  # single category
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    ap = float(coco_eval.stats[0]) if coco_eval.stats is not None else 0.0
    ap50 = float(coco_eval.stats[1]) if coco_eval.stats is not None else 0.0
    ap75 = float(coco_eval.stats[2]) if coco_eval.stats is not None else 0.0

    # ---------- Dataset Mean IoU (instance, unmatched GT=0) ----------
    dataset_mean_iou = (matches_iou_sum / gt_count_total) if gt_count_total > 0 else 0.0

    print("\n========== RESULTS (ROAD only) ==========")
    print(f"COCO segm mAP@[.5:.95]: {ap:.4f}")
    print(f"COCO segm AP@0.50    : {ap50:.4f}")
    print(f"COCO segm AP@0.75    : {ap75:.4f}")
    print(f"Dataset Mean IoU (Hungarian, unmatched GT=0): {dataset_mean_iou:.4f}")
    print("Done.")


if __name__ == "__main__":
    main()
