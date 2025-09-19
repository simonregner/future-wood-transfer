import os
import cv2
import numpy as np
import random
import hashlib
import json
from tqdm import tqdm

# =========================
# Configuration
# =========================
# Toggle: save panoptic & semseg mask PNGs or not
SAVE_MASK_IMAGES = False  # <-- set to True if you want mask images

ENABLE_AUGMENTATION = False  # set False to disable


# Each INPUT_ROOT must contain subfolders: train/, val/, test/
# and inside each split: images/  and  labels/  (tolerant to naming)
INPUT_ROOTS = [
    '/home/simon/Documents/Master-Thesis/data/yolo_training_data_road',
]

output_root = '/home/simon/Documents/Master-Thesis/data/coco_training_data_road'

# Build output folders conditionally
output_structure = {
    'train': {
        'images': os.path.join(output_root, 'train'),
        'panoptic': os.path.join(output_root, 'panoptic_train'),
        'semseg': os.path.join(output_root, 'panoptic_semseg_train')
    },
    'val': {
        'images': os.path.join(output_root, 'val'),
        'panoptic': os.path.join(output_root, 'panoptic_val'),
        'semseg': os.path.join(output_root, 'panoptic_semseg_val')
    },
    'test': {
        'images': os.path.join(output_root, 'test'),
        'panoptic': os.path.join(output_root, 'panoptic_test'),
        'semseg': os.path.join(output_root, 'panoptic_semseg_test')
    }
}

# Always ensure image dirs exist
for split in output_structure:
    os.makedirs(output_structure[split]['images'], exist_ok=True)
# Only create mask dirs if we will write masks
if SAVE_MASK_IMAGES:
    for split in output_structure:
        os.makedirs(output_structure[split]['panoptic'], exist_ok=True)
        os.makedirs(output_structure[split]['semseg'], exist_ok=True)

os.makedirs(os.path.join(output_root, 'annotations'), exist_ok=True)

# =========================
# Settings
# =========================
# Keep "background" at index 0 in your mapping; categories are 1-based in JSON.
# categories = [
#     "background",
#     "bugle_road",
#     "left_turn",
#     "right_turn",
#     "road",
#     "straight_turn",
#     "intersection",
#     "lane"
# ]

categories = [
    "road",
    "road_boundary"
]


# =========================
# Utilities
# =========================
IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

def generate_unique_name(path, suffix=""):
    fname = os.path.basename(path)
    fhash = hashlib.md5(path.encode()).hexdigest()[:8]
    base, ext = os.path.splitext(fname)
    return f"{base}_{fhash}{('_' + suffix) if suffix else ''}{ext}"

def parse_yolo_polygon(label_line, width, height):
    """
    YOLO polygon format per line:
      <class_id> x1 y1 x2 y2 ... (normalized 0..1)
    Needs at least 3 points (6 coords) -> 1 + 6 = 7 tokens.
    """
    parts = label_line.strip().split()
    if len(parts) < 7:
        return None, None
    try:
        class_id = int(float(parts[0]))
    except ValueError:
        return None, None
    coords = list(map(float, parts[1:]))

    polygon = []
    for i in range(0, len(coords), 2):
        x = int(round(coords[i] * width))
        y = int(round(coords[i + 1] * height))
        polygon.append([x, y])

    if len(polygon) < 3:
        return None, None
    return class_id, polygon

def bbox_coco(polygon):
    xs = [pt[0] for pt in polygon]
    ys = [pt[1] for pt in polygon]
    min_x, min_y = min(xs), min(ys)
    max_x, max_y = max(xs), max(ys)
    return [int(min_x), int(min_y), int(max_x - min_x), int(max_y - min_y)]

def area_calculator(polygon):
    # Shoelace formula
    return float(0.5 * abs(sum(
        polygon[i][0] * polygon[(i + 1) % len(polygon)][1] -
        polygon[(i + 1) % len(polygon)][0] * polygon[i][1]
        for i in range(len(polygon))
    )))

def creates_categories(categories_list):
    # COCO categories are 1-based; here background gets id=1
    return [{"id": i + 1, "name": cat, "supercategory": ""} for i, cat in enumerate(categories_list)]

def to_png_name(filename):
    base, _ = os.path.splitext(filename)
    return base + '.png'

# =========================
# Augmentations (photometric only)
# =========================
def adjust_brightness(image, factor):
    return cv2.convertScaleAbs(image, alpha=factor, beta=0)

def add_gaussian_noise(image, std_dev=5):
    noise = np.random.normal(0, std_dev, image.shape).astype(np.int16)
    noisy = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return noisy

def jitter_colors(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= random.uniform(0.5, 1.5)
    hsv[:, :, 2] *= random.uniform(0.5, 1.5)
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

augmentations = {
    'darker': lambda img: adjust_brightness(img, 0.5),
    'brighter': lambda img: adjust_brightness(img, 1.5),
    'noise': add_gaussian_noise,
    'jitter': jitter_colors
}

# =========================
# Data collection
# =========================
def _first_existing(path_candidates):
    for p in path_candidates:
        if os.path.isdir(p):
            return p
    return None

def collect_pairs_for_split(split_name):
    """
    For each INPUT_ROOT that has a subfolder named `split_name`,
    look for <split>/images and <split>/labels (tolerant naming),
    and pair files by basename.
    """
    pairs = []
    image_dir_names = ['images', 'image', 'imgs', 'img']
    label_dir_names = ['labels', 'label', 'annotations', 'ann']

    for root in INPUT_ROOTS:
        split_dir = os.path.join(root, split_name)
        if not os.path.isdir(split_dir):
            continue

        images_dir = _first_existing([os.path.join(split_dir, d) for d in image_dir_names])
        labels_dir = _first_existing([os.path.join(split_dir, d) for d in label_dir_names])
        if images_dir is None or labels_dir is None:
            continue

        for fname in os.listdir(images_dir):
            if not fname.lower().endswith(IMG_EXTS):
                continue
            base = os.path.splitext(fname)[0]
            label_path = os.path.join(labels_dir, base + '.txt')
            img_path = os.path.join(images_dir, fname)
            if os.path.exists(label_path):
                pairs.append((img_path, label_path))
    return pairs

split_data = {
    'train': collect_pairs_for_split('train'),
    'val': collect_pairs_for_split('val'),
    'test': collect_pairs_for_split('test'),
}

# =========================
# Main processing
# =========================
def process_and_save(pairs, split):
    image_id = 1
    annotation_id = 1
    segment_id = 1  # used only if saving panoptic colors

    coco_json = {
        "licenses": [],
        "info": {},
        "categories": creates_categories(categories),
        "images": [],
        "annotations": []
    }

    for img_path, label_path in tqdm(pairs, desc=f"Processing {split}"):
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Skipped (read error): {img_path}")
            continue
        h, w = img.shape[:2]
        base_name = generate_unique_name(img_path)

        # ---- Read & parse labels ONCE; skip empty/invalid ----
        polygons_by_class = {}
        try:
            with open(label_path, 'r') as f:
                lines = [ln for ln in f.readlines() if ln.strip()]
        except Exception as e:
            print(f"⚠️ Skipped (label read error): {label_path} ({e})")
            continue

        if len(lines) == 0:
            # Requirement: if txt is empty, do not copy image and do not add to JSON
            continue

        for line in lines:
            class_id, polygon = parse_yolo_polygon(line, w, h)
            if polygon is not None:
                polygons_by_class.setdefault(class_id, []).append(polygon)

        if sum(len(v) for v in polygons_by_class.values()) == 0:
            continue
        # -background-----------------------------------------------------

        def save_data(img_variant, variant_name, h, w, polygons_by_class):
            nonlocal image_id, annotation_id, segment_id

            out_img_path = os.path.join(output_structure[split]['images'], variant_name)
            cv2.imwrite(out_img_path, img_variant)

            # Prepare masks only if requested
            if SAVE_MASK_IMAGES:
                panoptic_mask = np.zeros((h, w, 3), dtype=np.uint8)
                semseg_mask = np.zeros((h, w), dtype=np.uint8)

            # Collect per-instance data for instances JSON
            segments_for_json = []

            for class_id, polygons in polygons_by_class.items():
                for polygon in polygons:
                    if len(polygon) < 3:
                        continue

                    # COCO polygon segmentation: flattened [x1,y1,x2,y2,...]
                    flat = []
                    for x, y in polygon:
                        flat.extend([float(x), float(y)])

                    # Compute bbox, area
                    bbox = bbox_coco(polygon)
                    area = area_calculator(polygon)

                    # Draw masks only if saving them
                    if SAVE_MASK_IMAGES:
                        color = [segment_id % 256, (segment_id >> 8) % 256, (segment_id >> 16) % 256]
                        poly_np = np.array([polygon], dtype=np.int32)
                        cv2.fillPoly(panoptic_mask, poly_np, color)
                        cv2.fillPoly(semseg_mask, poly_np, int(class_id))
                        segment_id += 1  # increment only when we used color id

                    segments_for_json.append({
                        "category_id": int(class_id),   # will +1 when writing
                        "area": float(area),
                        "bbox": [int(x) for x in bbox],
                        "segmentation": [flat],
                        "iscrowd": 0
                    })

            # Save masks (optional)
            pan_mask_name = None
            if SAVE_MASK_IMAGES:
                pan_mask_name = to_png_name(variant_name)
                cv2.imwrite(os.path.join(output_structure[split]['panoptic'], pan_mask_name), panoptic_mask)
                cv2.imwrite(os.path.join(output_structure[split]['semseg'], pan_mask_name), semseg_mask)

            # Add image entry
            coco_json["images"].append({
                "id": int(image_id),
                "width": int(w),
                "height": int(h),
                "file_name": os.path.basename(out_img_path)
            })

            # Add annotations; include mask filename only if masks were saved
            for seg in segments_for_json:
                ann = {
                    "id": int(annotation_id),
                    "image_id": int(image_id),
                    "category_id": int(seg["category_id"]) + 1,  # shift by +1 (background at 0)
                    "segmentation": seg["segmentation"],
                    "area": float(seg["area"]),
                    "bbox": [int(x) for x in seg["bbox"]],
                    "iscrowd": 0
                }
                if SAVE_MASK_IMAGES and pan_mask_name:
                    ann["file_name"] = pan_mask_name  # optional, non-standard; omit otherwise
                coco_json["annotations"].append(ann)
                annotation_id += 1

            image_id += 1

        # Save original
        save_data(img, base_name, h, w, polygons_by_class)

        # Save augmented (photometric only → polygons remain valid)
        if ENABLE_AUGMENTATION:
            num_augs = random.randint(1, 2)
            chosen_augs = random.sample(list(augmentations.keys()), num_augs)
            for aug_name in chosen_augs:
                aug_img = augmentations[aug_name](img)
                aug_base_name = generate_unique_name(img_path, suffix=aug_name)
                save_data(aug_img, aug_base_name, h, w, polygons_by_class)

    # Save COCO JSON for this split
    with open(os.path.join(output_root, f'annotations/instances_{split}.json'), 'w') as f:
        json.dump(coco_json, f, indent=2)

# =========================
# Run
# =========================
process_and_save(split_data['train'], 'train')
process_and_save(split_data['val'], 'val')
process_and_save(split_data['test'], 'test')

print(f"✅ Done. Wrote images and COCO instances JSON with polygon segmentations."
      f"{' Panoptic/semseg mask PNGs were saved.' if SAVE_MASK_IMAGES else ' Mask PNGs were NOT saved.'}")
