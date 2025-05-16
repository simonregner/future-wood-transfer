import os
import cv2
import numpy as np
import random
import hashlib
import json
from tqdm import tqdm

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
    '/home/simon/Documents/Master-Thesis/data/yolo_lanes_smoothed/MM_ForestRoads/images',
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

# Set flags
ENABLE_AUGMENTATION = False
TRAIN_SPLIT = 0.7

# Helper functions
def adjust_brightness(img, factor): return cv2.convertScaleAbs(img, alpha=factor, beta=0)
def add_gaussian_noise(img, std_dev):
    noise = np.random.normal(0, std_dev, img.shape).astype(np.uint8)
    return cv2.add(img, noise)
def jitter_colors(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.6, 1.4), 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * random.uniform(0.6, 1.4), 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
def flip_image(img): return cv2.flip(img, 1)

augmentations = {
    'darker': adjust_brightness,
    'brighter': lambda img: adjust_brightness(img, 1.5),
    'noise': lambda img: add_gaussian_noise(img, 5),
    'jitter': jitter_colors,
    'flip': flip_image,
}

def generate_unique_name(path, suffix=""):
    fname = os.path.basename(path)
    fhash = hashlib.md5(path.encode()).hexdigest()[:8]
    base, ext = os.path.splitext(fname)
    return f"{base}_{fhash}{('_' + suffix) if suffix else ''}{ext}"

# Collect image-label pairs
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
split_data = {
    'train': all_pairs[:split_idx],
    'val': all_pairs[split_idx:]
}

coco_output = {
    'train': {'images': [], 'annotations': [], 'categories': []},
    'val': {'images': [], 'annotations': [], 'categories': []}
}

annotation_id = 1
segment_id = 1

# Parse YOLO polygon
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

# Generate COCO annotations
def create_instance_annotations(polygons_by_class, image_id):
    global annotation_id
    annotations = []
    for class_id, polygons in polygons_by_class.items():
        for polygon in polygons:
            segmentation = [np.array(polygon).flatten().tolist()]
            if len(segmentation[0]) < 6:
                continue
            contour = np.array(polygon).reshape((-1, 1, 2)).astype(np.int32)
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            annotations.append({
                'id': annotation_id,
                'image_id': image_id,
                'category_id': int(class_id),
                'segmentation': segmentation,
                'area': float(area),
                'bbox': [x, y, w, h],
                'iscrowd': 0
            })
            annotation_id += 1
    return annotations

# Processing function
def process_and_save(pairs, split):
    global segment_id
    image_id = 1
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

        coco_output[split]['images'].append({
            'id': image_id,
            'file_name': os.path.basename(img_out),
            'height': h,
            'width': w
        })

        # Load label
        with open(label_path, 'r') as f:
            lines = f.readlines()
        polygons_by_class = {}
        for line in lines:
            class_id, polygon = parse_yolo_polygon(line, w, h)
            if polygon is not None:
                polygons_by_class.setdefault(class_id, []).append(polygon)

        # Create semantic and panoptic masks
        panoptic_mask = np.zeros((h, w, 3), dtype=np.uint8)
        semseg_mask = np.zeros((h, w), dtype=np.uint8)
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
                    "area": area,
                    "bbox": [x, y, w_box, h_box],
                    "iscrowd": 0
                })
                segment_id += 1

        # Save masks
        pan_mask_path = os.path.join(output_structure[split]['panoptic'], base_name.replace(".jpg", ".png"))
        semseg_mask_path = os.path.join(output_structure[split]['semseg'], base_name.replace(".jpg", ".png"))
        cv2.imwrite(pan_mask_path, panoptic_mask)
        cv2.imwrite(semseg_mask_path, semseg_mask)

        # Annotations
        coco_output[split]['annotations'].append({
            "image_id": image_id,
            "file_name": os.path.basename(pan_mask_path),
            "segments_info": segments_info
        })

        image_id += 1

# Add dummy categories
for i in range(1, 21):
    cat = {"id": i, "name": f"class_{i}"}
    coco_output['train']['categories'].append(cat)
    coco_output['val']['categories'].append(cat)

# Run
process_and_save(split_data['train'], 'train')
process_and_save(split_data['val'], 'val')

# Save annotations
with open(os.path.join(output_root, 'annotations/panoptic_train.json'), 'w') as f:
    json.dump(coco_output['train'], f)
with open(os.path.join(output_root, 'annotations/panoptic_val.json'), 'w') as f:
    json.dump(coco_output['val'], f)

print("✅ Mask2Former-compatible panoptic dataset generated.")
