import os
import cv2
import numpy as np
import random
import hashlib
import json
from tqdm import tqdm

# Configuration
input_folders = [
    '/home/simon/Documents/Master-Thesis/data/yolo_lanes_smoothed/GOOSE/images',
    '/home/simon/Documents/Master-Thesis/data/yolo_lanes_smoothed/ROSBAG_INTERSECTION/images',
    '/home/simon/Documents/Master-Thesis/data/yolo_lanes_smoothed/Intersections/images',
    '/home/simon/Documents/Master-Thesis/data/yolo_lanes_smoothed/Google_Maps/images',
    '/home/simon/Documents/Master-Thesis/data/yolo_lanes_smoothed/Google_Maps_MacBook/images',
    '/home/simon/Documents/Master-Thesis/data/yolo_lanes_smoothed/Forest_Testarea/images',
    '/home/simon/Documents/Master-Thesis/data/yolo_lanes_smoothed/MM_ForestRoads_01_1/images',
    '/home/simon/Documents/Master-Thesis/data/yolo_lanes_smoothed/MM_ForestRoads_01_2/images',
    '/home/simon/Documents/Master-Thesis/data/yolo_lanes_smoothed/MM_ForestRoads_02_1/images',
    '/home/simon/Documents/Master-Thesis/data/yolo_lanes_smoothed/MM_ForestRoads_02_2/images',
    '/home/simon/Documents/Master-Thesis/data/yolo_lanes_smoothed/MM_INTERSECTION_JAKOB/images',
    '/home/simon/Documents/Master-Thesis/data/yolo_lanes_smoothed/ROSBAG_UNI/images',
]

output_root = '/home/simon/Documents/Master-Thesis/data/coco_training_data'

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
    }
}
for split in output_structure:
    for key in output_structure[split]:
        os.makedirs(output_structure[split][key], exist_ok=True)
os.makedirs(os.path.join(output_root, 'annotations'), exist_ok=True)

TRAIN_SPLIT = 0.7
categories = [
    "background",
    "bugle_road",
    "left_turn",
    "right_turn",
    "road",
    "straight_turn",
    "intersection",
    "lane"
]

ENABLE_AUGMENTATION = True

# Utility functions
def generate_unique_name(path, suffix=""):
    fname = os.path.basename(path)
    fhash = hashlib.md5(path.encode()).hexdigest()[:8]
    base, ext = os.path.splitext(fname)
    return f"{base}_{fhash}{('_' + suffix) if suffix else ''}{ext}"

def parse_yolo_polygon(label_line, width, height):
    parts = label_line.strip().split()
    if len(parts) < 7: return None, None
    class_id = int(parts[0])
    coords = list(map(float, parts[1:]))
    polygon = []
    for i in range(0, len(coords), 2):
        x = int(coords[i] * width)
        y = int(coords[i + 1] * height)
        polygon.append([x, y])
    return class_id, polygon

def bbox_coco(polygon):
    x_coord = [pt[0] for pt in polygon]
    y_coord = [pt[1] for pt in polygon]
    min_x = min(x_coord)
    min_y = min(y_coord)
    max_x = max(x_coord)
    max_y = max(y_coord)
    return [min_x, min_y, max_x - min_x, max_y - min_y]

def area_calculator(polygon):
    return 0.5 * abs(sum(polygon[i][0] * polygon[(i + 1) % len(polygon)][1] -
                         polygon[(i + 1) % len(polygon)][0] * polygon[i][1]
                         for i in range(len(polygon))))

def creates_categories(categories):
    return [{"id": i+1, "name": cat, "supercategory": ""} for i, cat in enumerate(categories)]

# Augmentations
def adjust_brightness(image, factor):
    return cv2.convertScaleAbs(image, alpha=factor, beta=0)

def add_gaussian_noise(image, std_dev=5):
    noise = np.random.normal(0, std_dev, image.shape).astype(np.uint8)
    return cv2.add(image, noise)

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

# Collect data
all_pairs = []
for folder in input_folders:
    label_folder = os.path.join(os.path.dirname(folder), 'labels')
    for fname in os.listdir(folder):
        if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(folder, fname)
            label_path = os.path.join(label_folder, os.path.splitext(fname)[0] + '.txt')
            if os.path.exists(label_path):
                all_pairs.append((img_path, label_path))
random.shuffle(all_pairs)
split_idx = int(len(all_pairs) * TRAIN_SPLIT)
split_data = {'train': all_pairs[:split_idx], 'val': all_pairs[split_idx:]}

# Main processing
def process_and_save(pairs, split):
    image_id = 1
    annotation_id = 1
    segment_id = 1
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
            print(f"⚠️ Skipped: {img_path}")
            continue
        h, w = img.shape[:2]
        base_name = generate_unique_name(img_path)

        def save_data(img_variant, variant_name):
            nonlocal image_id, annotation_id, segment_id
            out_img_path = os.path.join(output_structure[split]['images'], variant_name)
            cv2.imwrite(out_img_path, img_variant)

            panoptic_mask = np.zeros((h, w, 3), dtype=np.uint8)
            semseg_mask = np.zeros((h, w), dtype=np.uint8)
            segments_info = []
            polygons_by_class = {}

            with open(label_path, 'r') as f:
                lines = f.readlines()
            for line in lines:
                class_id, polygon = parse_yolo_polygon(line, w, h)
                if polygon is not None:
                    polygons_by_class.setdefault(class_id, []).append(polygon)

            for class_id, polygons in polygons_by_class.items():
                for polygon in polygons:
                    color = [segment_id % 256, (segment_id >> 8) % 256, (segment_id >> 16) % 256]
                    poly_np = np.array([polygon], dtype=np.int32)
                    cv2.fillPoly(panoptic_mask, poly_np, color)
                    cv2.fillPoly(semseg_mask, poly_np, class_id)
                    bbox = bbox_coco(polygon)
                    area = area_calculator(polygon)
                    segments_info.append({
                        "id": segment_id,
                        "category_id": class_id,
                        "area": float(area),
                        "bbox": bbox,
                        "iscrowd": 0
                    })
                    segment_id += 1

            pan_mask_name = variant_name.replace(".jpg", ".png").replace(".jpeg", ".png")
            cv2.imwrite(os.path.join(output_structure[split]['panoptic'], pan_mask_name), panoptic_mask)
            cv2.imwrite(os.path.join(output_structure[split]['semseg'], pan_mask_name), semseg_mask)

            coco_json["images"].append({
                "id": image_id,
                "width": w,
                "height": h,
                "file_name": os.path.basename(out_img_path)
            })

            for seg in segments_info:
                coco_json["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": seg["category_id"] + 1,
                    "segmentation": [],  # optional: polygons
                    "area": seg["area"],
                    "bbox": seg["bbox"],
                    "iscrowd": 0,
                    "file_name": pan_mask_name
                })
                annotation_id += 1

            image_id += 1

        # Save original
        save_data(img, base_name)

        # Save augmented
        if ENABLE_AUGMENTATION:
            num_augs = random.randint(1, 2)
            chosen_augs = random.sample(list(augmentations.keys()), num_augs)
            for aug_name in chosen_augs:
                aug_img = augmentations[aug_name](img)
                aug_base_name = generate_unique_name(img_path, suffix=aug_name)
                save_data(aug_img, aug_base_name)

    # Save COCO JSON
    with open(os.path.join(output_root, f'annotations/instances_{split}.json'), 'w') as f:
        json.dump(coco_json, f, indent=2)

# Run
process_and_save(split_data['train'], 'train')
process_and_save(split_data['val'], 'val')

print("✅ All done! Images, masks, and COCO annotations are saved.")
