import os
import cv2
import numpy as np
import random
import shutil
import hashlib

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
        'images': os.path.join(output_root, 'train/images'),
        'masks': os.path.join(output_root, 'train/masks')
    },
    'val': {
        'images': os.path.join(output_root, 'val/images'),
        'masks': os.path.join(output_root, 'val/masks')
    }
}

os.makedirs(output_structure['train']['images'], exist_ok=True)
os.makedirs(output_structure['train']['masks'], exist_ok=True)
os.makedirs(output_structure['val']['images'], exist_ok=True)
os.makedirs(output_structure['val']['masks'], exist_ok=True)

# Set flags
ENABLE_AUGMENTATION = True
TRAIN_SPLIT = 1.0

# Augmentations
def adjust_brightness(img, factor): return cv2.convertScaleAbs(img, alpha=factor, beta=0)
def add_gaussian_noise(img, std_dev):
    noise = np.random.normal(0, std_dev, img.shape).astype(np.uint8)
    return cv2.add(img, noise)
def jitter_colors(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.6, 1.4), 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * random.uniform(0.6, 1.4), 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def flip_image_and_mask(image, mask):
    return cv2.flip(image, 1), cv2.flip(mask, 1)

augmentations = {
    'darker': lambda img, msk: (adjust_brightness(img, 0.5), msk),
    'brighter': lambda img, msk: (adjust_brightness(img, 1.5), msk),
    'noise': lambda img, msk: (add_gaussian_noise(img, 5), msk),
    'jitter': lambda img, msk: (jitter_colors(img), msk),
    'flip': lambda img, msk: flip_image_and_mask(img, msk),
}

def generate_unique_name(path, suffix=""):
    fname = os.path.basename(path)
    fhash = hashlib.md5(path.encode()).hexdigest()[:8]
    base, ext = os.path.splitext(fname)
    return f"{base}_{fhash}{('_' + suffix) if suffix else ''}{ext}"

# Collect image-mask pairs
all_pairs = []
for folder in input_folders:
    mask_folder = os.path.join(os.path.dirname(folder), 'masks')
    for fname in os.listdir(folder):
        if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(folder, fname)
            mask_path = os.path.join(mask_folder, os.path.splitext(fname)[0] + '.png')
            if os.path.exists(mask_path):
                all_pairs.append((img_path, mask_path))

# Shuffle and split
random.shuffle(all_pairs)
split_idx = int(len(all_pairs) * TRAIN_SPLIT)
split_data = {
    'train': all_pairs[:split_idx],
    'val': all_pairs[split_idx:]
}

# Process function
def process_and_save(pairs, split):
    print(f"\n🔄 Processing {split} set ({len(pairs)} samples)")
    for img_path, mask_path in tqdm(pairs, desc=f"{split.upper()} set"):
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            print(f"⚠️ Skipped: {img_path}")
            continue

        base_name = generate_unique_name(img_path)
        base_no_ext = os.path.splitext(base_name)[0]

        img_out = os.path.join(output_structure[split]['images'], base_name)
        mask_out = os.path.join(output_structure[split]['masks'], base_no_ext + '.png')
        cv2.imwrite(img_out, img)
        cv2.imwrite(mask_out, mask)

        # Augmentation with inner progress bar
        if ENABLE_AUGMENTATION:
            num_aug = random.randint(2, 4)
            aug_names = random.sample(list(augmentations.keys()), num_aug)
            for aug in aug_names:
                aug_img, aug_mask = augmentations[aug](img, mask)
                aug_name = generate_unique_name(img_path, suffix=aug)
                aug_base = os.path.splitext(aug_name)[0]
                cv2.imwrite(os.path.join(output_structure[split]['images'], aug_name), aug_img)
                cv2.imwrite(os.path.join(output_structure[split]['masks'], aug_base + '.png'), aug_mask)


# Run
process_and_save(split_data['train'], 'train')
process_and_save(split_data['val'], 'val')

print(f"✅ Done. Train: {len(split_data['train'])}, Val: {len(split_data['val'])}")
