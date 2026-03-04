import os
import cv2
import numpy as np

# --- CONFIG ---
image_folders = [
    '/home/simon/Documents/Master-Thesis/data/yolo_lanes_smoothed/CAVS/images',
    '/home/simon/Documents/Master-Thesis/data/yolo_lanes_smoothed/GOOSE/images',
    '/home/simon/Documents/Master-Thesis/data/yolo_lanes_smoothed/Road_Detection_Asphalt/images',
    '/home/simon/Documents/Master-Thesis/data/yolo_lanes_smoothed/ROSBAG_INTERSECTION/images',
    '/home/simon/Documents/Master-Thesis/data/yolo_lanes_smoothed/Intersections/images',
    '/home/simon/Documents/Master-Thesis/data/yolo_lanes_smoothed/Google_Maps/images',
    '/home/simon/Documents/Master-Thesis/data/yolo_lanes_smoothed/Google_Maps_MacBook/images',
    '/home/simon/Documents/Master-Thesis/data/yolo_lanes_smoothed/Forest_Testarea/images',
    '/home/simon/Documents/Master-Thesis/data/yolo_lanes_smoothed/MM_ForestRoads/images',
    '/home/simon/Documents/Master-Thesis/data/yolo_lanes_smoothed/MM_INTERSECTION_JAKOB/images',
    '/home/simon/Documents/Master-Thesis/data/yolo_lanes_smoothed/ROSBAG_UNI/images',
]
NUM_CLASSES = 8  # Adjust based on your dataset


def process_folder(image_dir):
    base_dir = os.path.dirname(image_dir)  # this gets the parent of /images
    label_dir = os.path.join(base_dir, "labels")
    mask_dir = os.path.join(base_dir, "masks")
    os.makedirs(mask_dir, exist_ok=True)

    for image_file in os.listdir(image_dir):
        if not image_file.lower().endswith((".jpg", ".png")):
            continue

        image_path = os.path.join(image_dir, image_file)
        label_path = os.path.join(label_dir, os.path.splitext(image_file)[0] + ".txt")
        mask_path = os.path.join(mask_dir, os.path.splitext(image_file)[0] + ".png")

        image = cv2.imread(image_path)
        if image is None:
            print(f"⚠️ Could not read image: {image_path}")
            continue

        height, width = image.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)

        if not os.path.exists(label_path):
            print(f"⚠️ Missing label for: {image_file}")
            continue

        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 7:
                    continue  # Must have at least 3 points (x, y pairs)

                class_id = int(parts[0])
                coords = list(map(float, parts[1:]))

                polygon = []
                for i in range(0, len(coords), 2):
                    x = int(coords[i] * width)
                    y = int(coords[i + 1] * height)
                    polygon.append([x, y])

                polygon_np = np.array([polygon], dtype=np.int32)
                cv2.fillPoly(mask, polygon_np, class_id)

        cv2.imwrite(mask_path, mask)
        print(f"✅ Saved: {mask_path}")

# --- PROCESS ALL IMAGE FOLDERS ---
for folder in image_folders:
    print(f"\n📂 Processing folder: {folder}")
    process_folder(folder)