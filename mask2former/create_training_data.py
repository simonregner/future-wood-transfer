import os
import cv2
import numpy as np
import random
import hashlib
import json
from tqdm import tqdm
from PIL import Image

# Configuration
input_folders = [
    #'/home/simon/Documents/Master-Thesis/data/yolo_lanes_smoothed/CAVS/images',
    #'/home/simon/Documents/Master-Thesis/data/yolo_lanes_smoothed/GOOSE/images',
    #'/home/simon/Documents/Master-Thesis/data/yolo_lanes_smoothed/Road_Detection_Asphalt/images',
    #'/home/simon/Documents/Master-Thesis/data/yolo_lanes_smoothed/ROSBAG_INTERSECTION/images',
    #'/home/simon/Documents/Master-Thesis/data/yolo_lanes_smoothed/Intersections/images',
    #'/home/simon/Documents/Master-Thesis/data/yolo_lanes_smoothed/Google_Maps/images',
    #'/home/simon/Documents/Master-Thesis/data/yolo_lanes_smoothed/Google_Maps_MacBook/images',
    '/home/simon/Documents/Master-Thesis/data/yolo_lanes_smoothed/Forest_Testarea/images',
    '/home/simon/Documents/Master-Thesis/data/yolo_lanes_smoothed/MM_ForestRoads_01_1/images',
    '/home/simon/Documents/Master-Thesis/data/yolo_lanes_smoothed/MM_INTERSECTION_JAKOB/images',
    #'/home/simon/Documents/Master-Thesis/data/yolo_lanes_smoothed/ROSBAG_UNI/images',
]

output_root = '/home/simon/Documents/Master-Thesis/data/yolo_training_data'  # base output folder

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

# --- UTILITY FUNCTIONS ---
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
    width = max_x - min_x
    height = max_y - min_y
    return [min_x, min_y, width, height]

def area_calculator(polygon):
    number_of_vertices = len(polygon)
    sum_1 = 0.0
    sum_2 = 0.0
    for i in range(number_of_vertices - 1):
        sum_1 += polygon[i][0] * polygon[i + 1][1]
        sum_2 += polygon[i][1] * polygon[i + 1][0]
    final_sum_1 = sum_1 + polygon[-1][0] * polygon[0][1]
    final_sum_2 = sum_2 + polygon[0][0] * polygon[-1][1]
    return abs((final_sum_1 - final_sum_2)) / 2

def creates_categories(categories):
    return [{"id": i+1, "name": cat, "supercategory": ""} for i, cat in enumerate(categories)]

# --- COLLECT IMAGE-LABEL PAIRS ---
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

# --- MAIN PROCESSING: COPY, GENERATE MASKS, BUILD COCO JSON ---
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

        # Save image
        img_out = os.path.join(output_structure[split]['images'], base_name)
        cv2.imwrite(img_out, img)

        # Create masks
        panoptic_mask = np.zeros((h, w, 3), dtype=np.uint8)
        semseg_mask = np.zeros((h, w), dtype=np.uint8)
        polygons_by_class = {}

        # Load label and parse polygons
        with open(label_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            class_id, polygon = parse_yolo_polygon(line, w, h)
            if polygon is not None:
                polygons_by_class.setdefault(class_id, []).append(polygon)

        segments_info = []
        for class_id, polygons in polygons_by_class.items():
            for polygon in polygons:
                color = [segment_id % 256, (segment_id >> 8) % 256, (segment_id >> 16) % 256]
                poly_np = np.array([polygon], dtype=np.int32)
                cv2.fillPoly(panoptic_mask, poly_np, color)
                cv2.fillPoly(semseg_mask, poly_np, class_id)
                x, y, w_box, h_box = cv2.boundingRect(poly_np)
                area = cv2.contourArea(poly_np)
                segments_info.append({
                    "id": segment_id,
                    "category_id": class_id,
                    "area": float(area),
                    "bbox": [x, y, w_box, h_box],
                    "iscrowd": 0
                })
                segment_id += 1

        # Save masks
        pan_mask_name = base_name.replace(".jpg", ".png").replace(".jpeg", ".png")
        pan_mask_path = os.path.join(output_structure[split]['panoptic'], pan_mask_name)
        semseg_mask_path = os.path.join(output_structure[split]['semseg'], pan_mask_name)
        cv2.imwrite(pan_mask_path, panoptic_mask)
        cv2.imwrite(semseg_mask_path, semseg_mask)

        # Save image entry
        coco_json["images"].append({
            "id": image_id,
            "width": w,
            "height": h,
            "file_name": os.path.basename(img_out)
        })

        # Save annotation entry: one entry per mask
        # You can choose: one annotation per instance, or per image (panoptic-style)
        for class_id, polygons in polygons_by_class.items():
            for polygon in polygons:
                if len(polygon) < 3:
                    continue
                coco_json["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id + 1,  # 1-based
                    "segmentation": [[x for pt in polygon for x in pt]],
                    "area": area_calculator(polygon),
                    "bbox": bbox_coco(polygon),
                    "iscrowd": 0,
                    "file_name": pan_mask_name  # <- The mask image filename!
                })
                annotation_id += 1

        image_id += 1

    # Save the COCO JSON file
    with open(os.path.join(output_root, f'annotations/instances_{split}.json'), 'w') as f:
        json.dump(coco_json, f, indent=2)

# --- RUN ---
process_and_save(split_data['train'], 'train')
process_and_save(split_data['val'], 'val')

print("✅ All done! Images and masks are saved. COCO annotation includes the mask filenames.")