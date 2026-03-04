import os
import glob
import json
import random
import shutil
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
import cv2

# ----------- USER CONFIG HERE -----------
IMAGE_FOLDERS = [
    '/home/simon/Documents/Master-Thesis/data/yolo_lanes_smoothed/CAVS/images',
    '/home/simon/Documents/Master-Thesis/data/yolo_lanes_smoothed/GOOSE/images',
    #'/home/simon/Documents/Master-Thesis/data/yolo_lanes_smoothed/Road_Detection_Asphalt/images',
    '/home/simon/Documents/Master-Thesis/data/yolo_lanes_smoothed/ROSBAG_INTERSECTION/images',
    '/home/simon/Documents/Master-Thesis/data/yolo_lanes_smoothed/Intersections/images',
    '/home/simon/Documents/Master-Thesis/data/yolo_lanes_smoothed/Google_Maps/images',
    '/home/simon/Documents/Master-Thesis/data/yolo_lanes_smoothed/Google_Maps_MacBook/images',
    '/home/simon/Documents/Master-Thesis/data/yolo_lanes_smoothed/Forest_Testarea/images',
    '/home/simon/Documents/Master-Thesis/data/yolo_lanes_smoothed/MM_INTERSECTION_JAKOB/images',
    '/home/simon/Documents/Master-Thesis/data/yolo_lanes_smoothed/MM_ForestRoads_01_1/images',
    '/home/simon/Documents/Master-Thesis/data/yolo_lanes_smoothed/MM_ForestRoads_01_2/images',
    '/home/simon/Documents/Master-Thesis/data/yolo_lanes_smoothed/MM_ForestRoads_02_1/images',
    '/home/simon/Documents/Master-Thesis/data/yolo_lanes_smoothed/MM_ForestRoads_02_2/images',
    '/home/simon/Documents/Master-Thesis/data/yolo_lanes_smoothed/ROSBAG_UNI/images',
]
CATEGORIES = [
    "background",
    "bugle_road",
    "left_turn",
    "right_turn",
    "road",
    "straight_turn",
    "intersection",
    "lane"
]
TRAIN_SPLIT = 0.9  # 80% train, 20% val
OUTPUT_DIR = '/home/simon/Documents/Master-Thesis/data/coco_training_data'
SEED = 42  # For reproducibility
# ----------------------------------------

random.seed(SEED)

def parse_yolo_polygon_line(line):
    tokens = line.strip().split()
    class_id = int(tokens[0])
    coords = list(map(float, tokens[1:]))
    points = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
    return class_id, points

def get_category_dicts():
    # COCO expects category_id starting from 1
    return [
        {
            "id": i + 1,
            "name": name,
            "supercategory": "thing"
        }
        for i, name in enumerate(CATEGORIES)
    ]

def convert_polygon_to_coco(points, width, height):
    # scales normalized [0,1] points to image size, returns a flat list
    poly = []
    for x, y in points:
        px = min(max(x * width, 0), width-1)
        py = min(max(y * height, 0), height-1)
        poly.append(px)
        poly.append(py)
    return poly

def bbox_from_poly(poly):
    xs = poly[::2]
    ys = poly[1::2]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return [x_min, y_min, x_max - x_min, y_max - y_min]

def area_from_poly(poly):
    # Shoelace formula
    xs = poly[::2]
    ys = poly[1::2]
    n = len(xs)
    area = 0.0
    for i in range(n):
        area += xs[i] * ys[(i+1)%n] - xs[(i+1)%n] * ys[i]
    return abs(area) / 2.0

def sanitize_folder_name(folder_path):
    return os.path.basename(os.path.normpath(folder_path)).replace(' ', '_')

def main():
    # Gather images
    all_images = []
    for folder in IMAGE_FOLDERS:
        imgs = sorted(glob.glob(os.path.join(folder, "*.[jp][pn]g")))
        for img in imgs:
            all_images.append((img, folder))
    random.shuffle(all_images)
    n_train = int(len(all_images) * TRAIN_SPLIT)
    splits = {'train': all_images[:n_train], 'val': all_images[n_train:]}

    # Prepare output folders
    for split in ['train', 'val']:
        os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "annotations"), exist_ok=True)

    # Build category dict
    category_dicts = get_category_dicts()

    # Build a mapping to ensure image names are unique
    used_img_names = set()
    def unique_img_name(img_path, folder):
        base_folder = sanitize_folder_name(folder)
        img_base = os.path.basename(img_path)
        new_name = f"{base_folder}_{img_base}"
        # Make sure it is unique, even if multiple levels/folders could produce collisions
        orig_new_name = new_name
        i = 1
        while new_name in used_img_names:
            new_name = f"{base_folder}_{i}_{img_base}"
            i += 1
        used_img_names.add(new_name)
        return new_name

    # Process each split
    for split in ['train', 'val']:
        image_dicts = []
        ann_dicts = []
        ann_id = 1
        for img_id, (img_path, folder) in enumerate(tqdm(splits[split], desc=f"Processing {split}")):
            img_name = unique_img_name(img_path, folder)
            out_img_path = os.path.join(OUTPUT_DIR, split, img_name)
            shutil.copy(img_path, out_img_path)

            img = Image.open(img_path)
            width, height = img.size
            image_dicts.append({
                "file_name": img_name,
                "id": img_id,
                "width": width,
                "height": height,
            })

            # Find label
            label_name = os.path.splitext(os.path.basename(img_path))[0] + '.txt'
            label_path = os.path.join(os.path.dirname(folder), 'labels', label_name)
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                for line in lines:
                    class_id, points = parse_yolo_polygon_line(line)
                    poly = convert_polygon_to_coco(points, width, height)
                    bbox = bbox_from_poly(poly)
                    area = area_from_poly(poly)
                    ann_dicts.append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": class_id + 1,  # COCO IDs start from 1
                        "segmentation": [poly],
                        "bbox": [float(b) for b in bbox],
                        "area": float(area),
                        "iscrowd": 0
                    })
                    ann_id += 1

        # Write COCO instances JSON
        coco_json = {
            "images": image_dicts,
            "annotations": ann_dicts,
            "categories": category_dicts,
        }
        json_path = os.path.join(OUTPUT_DIR, "annotations", f"instances_{split}.json")
        with open(json_path, "w") as f:
            json.dump(coco_json, f, indent=2)

if __name__ == '__main__':
    main()