#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
YOLO (polygon) -> COCO Instance + Panoptic converter
STREAMING-WRITE: Images & Annotations werden sofort in JSON geschrieben.

Verzeichnislayout (YOLO):
  <yolo_root>/{train,val,test}/images/*.jpg|.png
  <yolo_root>/{train,val,test}/labels/*.txt

Liest Klassennamen aus:
  - <yolo_root>/data.yaml  (names: list ODER {id: name})
  - <yolo_root>/classes.txt (Fallback)
"""

import argparse
import json
import gc
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageOps
import yaml
from tqdm import tqdm


# ---------------------------
# JSON Streaming Writer
# ---------------------------

class StreamingCocoWriter:
    """Schreibt COCO-Instance-JSON inkrementell (images/annotations)."""
    def __init__(self, path: Path, info: Dict, categories: List[Dict]):
        self.f = open(path, "w", encoding="utf-8")
        self.path = path
        self.categories = categories
        self.first_image = True
        self.first_ann = True
        # Header + images array öffnen
        self.f.write('{"info":')
        json.dump(info, self.f)
        self.f.write(',"licenses":[],"images":[')

    def add_image(self, image_rec: Dict):
        if not self.first_image:
            self.f.write(",")
        json.dump(image_rec, self.f)
        self.first_image = False

    def open_annotations(self):
        # images array schließen + annotations array öffnen
        self.f.write('],"annotations":[')

    def add_annotation(self, ann_rec: Dict):
        if not hasattr(self, "_ann_opened"):
            self._ann_opened = True
        if not self.first_ann:
            self.f.write(",")
        json.dump(ann_rec, self.f)
        self.first_ann = False

    def close(self):
        # annotations array schließen + categories + JSON Ende
        if not hasattr(self, "_ann_opened"):
            # Falls nie eine Annotation kam, Annotations-Array trotzdem schließen
            self.f.write('],"annotations":[')
        self.f.write('],"categories":')
        json.dump(self.categories, self.f)
        self.f.write("}")
        self.f.close()


class StreamingPanopticWriter:
    """Schreibt COCO-Panoptic-JSON inkrementell (images/annotations)."""
    def __init__(self, path: Path, info: Dict, categories: List[Dict]):
        self.f = open(path, "w", encoding="utf-8")
        self.path = path
        self.categories = categories
        self.first_image = True
        self.first_ann = True
        self.f.write('{"info":')
        json.dump(info, self.f)
        self.f.write(',"images":[')

    def add_image(self, image_rec: Dict):
        if not self.first_image:
            self.f.write(",")
        json.dump(image_rec, self.f)
        self.first_image = False

    def open_annotations(self):
        self.f.write('],"annotations":[')

    def add_panoptic_annotation(self, ann_rec: Dict):
        if not hasattr(self, "_ann_opened"):
            self._ann_opened = True
        if not self.first_ann:
            self.f.write(",")
        json.dump(ann_rec, self.f)
        self.first_ann = False

    def close(self):
        if not hasattr(self, "_ann_opened"):
            self.f.write('],"annotations":[')
        self.f.write('],"categories":')
        json.dump(self.categories, self.f)
        self.f.write("}")
        self.f.close()


# ---------------------------
# Helpers
# ---------------------------

def load_class_names(yolo_root: Path) -> List[str]:
    yaml_path = yolo_root / "data.yaml"
    if yaml_path.exists():
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        names = data.get("names", None)
        if isinstance(names, dict):
            return [name for _, name in sorted(names.items(), key=lambda kv: int(kv[0]))]
        if isinstance(names, list):
            return names
    classes_path = yolo_root / "classes.txt"
    if classes_path.exists():
        with open(classes_path, "r") as f:
            return [ln.strip() for ln in f if ln.strip()]
    raise RuntimeError("No class names found. Provide data.yaml or classes.txt at the YOLO root.")

def list_images(img_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return sorted([p for p in img_dir.rglob("*") if p.suffix.lower() in exts])

def yolo_poly_txt_to_instances(txt_path: Path, img_w: int, img_h: int) -> List[Dict]:
    instances = []
    if not txt_path.exists():
        return instances
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            cls = int(float(parts[0]))
            coords = list(map(float, parts[1:]))
            if len(coords) % 2 != 0 or len(coords) < 6:
                continue
            xy_abs = []
            for i in range(0, len(coords), 2):
                x = coords[i] * img_w
                y = coords[i+1] * img_h
                xy_abs.append((float(x), float(y)))
            instances.append({"class_id": cls, "polygon_abs": xy_abs})
    return instances

def polygon_to_bbox_and_area(poly_xy: List[Tuple[float,float]]) -> Tuple[List[float], float]:
    xs = [p[0] for p in poly_xy]
    ys = [p[1] for p in poly_xy]
    x0, y0 = min(xs), min(ys)
    x1, y1 = max(xs), max(ys)
    bbox = [float(x0), float(y0), float(x1 - x0), float(y1 - y0)]
    area = 0.0
    for i in range(len(poly_xy)):
        x_i, y_i = poly_xy[i]
        x_j, y_j = poly_xy[(i+1) % len(poly_xy)]
        area += x_i * y_j - x_j * y_i
    area = abs(area) / 2.0
    return bbox, area

def encode_id_to_rgb(seg_id: int) -> Tuple[int,int,int]:
    r = seg_id % 256
    g = (seg_id // 256) % 256
    b = (seg_id // (256*256)) % 256
    return (r, g, b)

def draw_polygon_mask(size_wh: Tuple[int,int], polygon: List[Tuple[float,float]]) -> Image.Image:
    w, h = size_wh
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).polygon(polygon, outline=1, fill=1)
    return mask


# ---------------------------
# Core conversion
# ---------------------------

def process_split(
    split_name: str,
    yolo_root: Path,
    out_root: Path,
    categories: List[Dict],
    dataset_name_prefix: str
):
    img_dir = yolo_root / split_name / "images"
    lbl_dir = yolo_root / split_name / "labels"
    if not img_dir.exists():
        print(f"[WARN] Missing images dir: {img_dir} — skipping '{split_name}'.")
        return

    images = list_images(img_dir)
    if not images:
        print(f"[WARN] No images found in {img_dir} — skipping '{split_name}'.")
        return

    # Output
    inst_json_path = out_root / f"instances_{split_name}.json"
    panoptic_dir = out_root / f"panoptic_{split_name}"
    panoptic_dir.mkdir(parents=True, exist_ok=True)
    panoptic_json_path = out_root / f"panoptic_{split_name}.json"
    masks_root = out_root / f"masks_{split_name}"
    masks_root.mkdir(parents=True, exist_ok=True)

    # Streaming-Writer öffnen
    inst_writer = StreamingCocoWriter(
        inst_json_path,
        info={"description": f"{dataset_name_prefix} {split_name} (YOLO->COCO)", "version": "1.0", "year": 2025},
        categories=categories
    )
    pano_writer = StreamingPanopticWriter(
        panoptic_json_path,
        info={"description": f"{dataset_name_prefix} {split_name} panoptic", "version": "1.0", "year": 2025},
        categories=categories
    )

    ann_id_counter = 1
    panoptic_seg_id_counter = 1

    # Images-Arrays beider JSONs zuerst befüllen; wir schreiben pro Bild unmittelbar
    pbar = tqdm(images, desc=f"{split_name}: images", unit="img", leave=True)
    for img_idx, img_path in enumerate(pbar):
        # Bild öffnen
        img = Image.open(img_path)
        try:
            img = ImageOps.exif_transpose(img)
            w, h = img.size
        finally:
            # falls exif_transpose ein neues Objekt erzeugt, altes schließen
            pass

        rel_img_path = img_path.relative_to(yolo_root)
        image_id = img_idx + 1
        img_rec = {"id": image_id, "file_name": str(rel_img_path.as_posix()), "width": w, "height": h}

        # direkt in beide JSONs schreiben
        inst_writer.add_image(img_rec)
        pano_writer.add_image(img_rec)

        # Labels lesen
        txt_path = lbl_dir / (img_path.stem + ".txt")
        instances = yolo_poly_txt_to_instances(txt_path, w, h)
        pbar.set_postfix_str(f"anns={len(instances)}")

        # Panoptic Canvas
        pano_img = Image.new("RGB", (w, h), (0, 0, 0))
        pano_draw = pano_img.load()
        segments_info = []

        # Per-Instanz-Masken
        inst_mask_dir = masks_root / img_path.stem
        inst_mask_dir.mkdir(parents=True, exist_ok=True)

        # Ab hier Annotationen-Array öffnen (einmalig vor der ersten Ann)
        if ann_id_counter == 1:
            inst_writer.open_annotations()
        if panoptic_seg_id_counter == 1:
            pano_writer.open_annotations()

        for inst in instances:
            class_id = int(inst["class_id"])
            # polygon -> binäre Maske
            poly_xy = inst["polygon_abs"]
            poly_mask = draw_polygon_mask((w, h), poly_xy)

            # Instance-Annotation sofort in JSON
            bbox, _ = polygon_to_bbox_and_area(poly_xy)
            area_raster = int(np.array(poly_mask, dtype=np.uint8).sum())

            inst_writer.add_annotation({
                "id": ann_id_counter,
                "image_id": image_id,
                "category_id": class_id,
                "segmentation": [list(np.array(poly_xy).flatten())],
                "bbox": [float(b) for b in bbox],
                "area": float(area_raster),
                "iscrowd": 0
            })

            # Save per-instance mask
            mask_path = inst_mask_dir / f"inst_{ann_id_counter}.png"
            # 0/255 Maske
            poly_mask.point(lambda p: 255 if p > 0 else 0).convert("L").save(mask_path)

            ann_id_counter += 1

            # Panoptic: einfärben
            seg_id = panoptic_seg_id_counter
            panoptic_seg_id_counter += 1
            rgb = encode_id_to_rgb(seg_id)
            mask_np = np.array(poly_mask, dtype=np.uint8)
            ys, xs = np.where(mask_np > 0)
            for y, x in zip(ys, xs):
                pano_draw[x, y] = rgb

            if ys.size > 0:
                y0, y1 = int(ys.min()), int(ys.max())
                x0, x1 = int(xs.min()), int(xs.max())
                bbox_pan = [int(x0), int(y0), int(x1 - x0 + 1), int(y1 - y0 + 1)]
                area_pan = int(mask_np.sum())
            else:
                bbox_pan = [0, 0, 0, 0]
                area_pan = 0

            segments_info.append({
                "id": int(seg_id),
                "category_id": int(class_id),
                "area": int(area_pan),
                "bbox": [int(v) for v in bbox_pan],
                "iscrowd": 0
            })

            # Speicher bereinigen
            del mask_np, poly_mask
            gc.collect()

        # Panoptic-PNG + Annotations-Eintrag sofort schreiben
        pano_name = f"{img_path.stem}.png"
        pano_img.save(panoptic_dir / pano_name)
        pano_img.close()
        pano_writer.add_panoptic_annotation({
            "image_id": image_id,
            "file_name": pano_name,
            "segments_info": segments_info
        })

        # Bild-Objekt freigeben
        img.close()
        del segments_info
        gc.collect()

    # Writer sauber schließen
    pano_writer.close()
    inst_writer.close()

    print(f"[OK] {split_name}: instances -> {inst_json_path.name}, panoptic -> {panoptic_json_path.name}")


def build_categories(names: List[str], stuff_ids: Optional[List[int]] = None) -> List[Dict]:
    cats = []
    stuff_ids = set(stuff_ids or [])
    for cid, name in enumerate(names):
        cats.append({
            "id": cid,
            "name": name,
            "supercategory": "none",
            "isthing": 0 if cid in stuff_ids else 1
        })
    return cats


# ---------------------------
# CLI
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="YOLO polygon -> COCO Instance + Panoptic (streaming JSON write)")
    parser.add_argument("--yolo-root", type=str, required=True, help="YOLO dataset root with train/val/test")
    parser.add_argument("--out-root", type=str, required=True, help="Output root for COCO artifacts")
    parser.add_argument("--dataset-name-prefix", type=str, default="mydataset",
                        help="Name prefix used in JSON info / registration")
    parser.add_argument("--stuff-ids", type=int, nargs="*", default=None,
                        help="Class IDs to mark as 'stuff' (isthing=0)")
    parser.add_argument("--splits", type=str, nargs="*", default=["train", "val", "test"],
                        help="Which splits to convert")
    args = parser.parse_args()

    yolo_root = Path(args.yolo_root).resolve()
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    names = load_class_names(yolo_root)
    categories = build_categories(names, args.stuff_ids)

    print(f"Classes ({len(names)}): {names}")

    for split in tqdm(args.splits, desc="splits", unit="split", leave=True):
        process_split(split, yolo_root, out_root, categories, args.dataset_name_prefix)

    print("Done.")

if __name__ == "__main__":
    main()
