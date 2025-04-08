import os
import random
import shutil
import hashlib
import cv2
import numpy as np
from pathlib import Path

# Toggle augmentation on/off
ENABLE_AUGMENTATION = True

# New boolean: if True, flip only if one of the specific classes is in the label file.
FLIP_ONLY_IF_SPECIFIC_CLASSES = True
SPECIFIC_CLASSES = ['2', '3', '5']  # Flip only if one of these classes is present.

# Set main input folders
input_folders = [
    '/home/simon/Documents/Master-Thesis/data/yolo_lanes_smoothed/CAVS/images',
    '/home/simon/Documents/Master-Thesis/data/yolo_lanes_smoothed/GOOSE/images',
    #'/home/simon/Documents/Master-Thesis/data/yolo_lanes_smoothed/Road_Detection_Asphalt/images',
    #'/home/simon/Documents/Master-Thesis/data/yolo_lanes_smoothed/ROSBAG_INTERSECTION/images',
    '/home/simon/Documents/Master-Thesis/data/yolo_lanes_smoothed/Intersections/images',
    '/home/simon/Documents/Master-Thesis/data/yolo_lanes_smoothed/Google_Maps/images',
    '/home/simon/Documents/Master-Thesis/data/yolo_lanes_smoothed/Google_Maps_MacBook/images',
    '/home/simon/Documents/Master-Thesis/data/yolo_lanes_smoothed/Forest_Testarea/images',
    #'/home/simon/Documents/Master-Thesis/data/yolo_lanes_smoothed/ROSBAG_UNI/images',
]

# Define special folder and its labels folder (adjust paths as needed)
special_folder_images = '/home/simon/Documents/Master-Thesis/data/COCO/train2017'
special_folder_labels = '/home/simon/Documents/Master-Thesis/data/COCO/annotations/empty'
special_percentage = 0.1  # e.g., 10%

# Set output folder and structure
output_folder = '/home/simon/Documents/Master-Thesis/data/yolo_training_data'
output_structure = {
    'train': {'images': os.path.join(output_folder, 'train/images'),
              'labels': os.path.join(output_folder, 'train/labels')},
    'val': {'images': os.path.join(output_folder, 'val/images'),
            'labels': os.path.join(output_folder, 'val/labels')}
}

# Create output directories
for set_name, paths in output_structure.items():
    os.makedirs(paths['images'], exist_ok=True)
    os.makedirs(paths['labels'], exist_ok=True)

# Collect images and labels from main input folders
main_images = []
main_labels = []
for folder in input_folders:
    images = [os.path.join(folder, f) for f in os.listdir(folder)
              if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    labels = [os.path.join(folder, '../labels', os.path.splitext(f)[0] + '.txt')
              for f in os.listdir(folder)
              if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    main_images.extend(images)
    main_labels.extend(labels)

# Collect images and labels from the special folder
special_images = [os.path.join(special_folder_images, f) for f in os.listdir(special_folder_images)
                  if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
special_labels = [os.path.join(special_folder_labels, os.path.splitext(f)[0] + '.txt')
                  for f in os.listdir(special_folder_images)
                  if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

# Compute how many special images to add.
M = len(main_images)
desired_special_count = int((special_percentage * M) / (1 - special_percentage))
desired_special_count = min(desired_special_count, len(special_images))

print("Number of images: {}".format(M))
print("Number of Background images: {}".format(desired_special_count))

# Randomly sample from the special folder
indices = list(range(len(special_images)))
random.shuffle(indices)
selected_special_images = [special_images[i] for i in indices[:desired_special_count]]
selected_special_labels = [special_labels[i] for i in indices[:desired_special_count]]

# Combine main and selected special images and labels
all_images = main_images + selected_special_images
all_labels = main_labels + selected_special_labels

# Shuffle images and labels together (maintaining their pairing)
indices = list(range(len(all_images)))
random.shuffle(indices)
all_images = [all_images[i] for i in indices]
all_labels = [all_labels[i] for i in indices]

# Calculate dataset split indices
num_images = len(all_images)
train_size = int(num_images * 0.7)
train_images, val_images = all_images[:train_size], all_images[train_size:]
train_labels, val_labels = all_labels[:train_size], all_labels[train_size:]


# Helper function to generate unique file names
def generate_unique_name(file_path, suffix=""):
    file_name = os.path.basename(file_path)
    folder_name = os.path.basename(os.path.dirname(file_path))
    unique_hash = hashlib.md5((folder_name + file_name).encode()).hexdigest()[:8]
    base, ext = os.path.splitext(file_name)
    return f"{base}_{unique_hash}{('_' + suffix) if suffix else ''}{ext}"


# Augmentation functions
def adjust_brightness(image, factor):
    return cv2.convertScaleAbs(image, alpha=factor, beta=0)


def add_gaussian_noise(image, std_dev):
    noise = np.random.normal(0, std_dev, image.shape).astype(np.uint8)
    return cv2.add(image, noise)


def jitter_colors(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_image[:, :, 2] = np.clip(hsv_image[:, :, 2] * random.uniform(0.5, 1.7), 0, 255)
    hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * random.uniform(0.5, 1.7), 0, 255)
    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

# New function to flip image and update labels (including flipping polygons)
def clamp(value, min_value=0.0, max_value=1.0):
    return max(min_value, min(max_value, value))

def flip_image_and_update_labels(image, label_path, output_image_path, output_label_path):
    flipped_image = cv2.flip(image, 1)
    cv2.imwrite(output_image_path, flipped_image)

    if os.path.exists(label_path):
        with open(label_path, 'r') as file:
            lines = file.readlines()

        flipped_annotations = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 3 or len(parts[1:]) % 2 != 0:
                print(f"Malformed annotation skipped: {line.strip()}")
                continue

            cls = parts[0]

            # Swap class indices 2 and 3 if needed
            cls = '3' if cls == '2' else '2' if cls == '3' else cls

            coords = [float(coord) for coord in parts[1:]]
            flipped_coords = []

            for i in range(0, len(coords), 2):
                orig_x = coords[i]
                orig_y = coords[i + 1]

                # Flip x coordinate
                flipped_x = clamp(1.0 - orig_x)
                flipped_y = clamp(orig_y)  # y stays same

                flipped_coords.extend([f"{flipped_x:.6f}", f"{flipped_y:.6f}"])

            new_line = f"{cls} " + " ".join(flipped_coords) + "\n"
            flipped_annotations.append(new_line)

        with open(output_label_path, 'w') as file:
            file.writelines(flipped_annotations)
    else:
        open(output_label_path, 'w').close()

# Mapping augmentation names to functions.
# For flip, we now call our custom function.
augmentations = {
    'darker': lambda img: adjust_brightness(img, 0.5),
    'brighter': lambda img: adjust_brightness(img, 1.5),
    'noise': lambda img: add_gaussian_noise(img, 1.5),
    'jitter': lambda img: jitter_colors(img),
    # 'flip' will be handled separately using our custom function.
}


# Helper function to copy files and apply augmentations
def copy_and_augment_files(images, labels, set_name):
    for image_path, label_path in zip(images, labels):
        # Generate unique file names for original image and label
        unique_image_name = generate_unique_name(image_path)
        output_image_path = os.path.join(output_structure[set_name]['images'], unique_image_name)
        unique_label_name = os.path.splitext(unique_image_name)[0] + '.txt'
        output_label_path = os.path.join(output_structure[set_name]['labels'], unique_label_name)

        # Copy original image
        shutil.copy(image_path, output_image_path)

        # Copy label if exists; otherwise, create an empty label file
        if os.path.exists(label_path):
            shutil.copy(label_path, output_label_path)
        else:
            open(output_label_path, 'w').close()

        # Apply augmentations if enabled
        if ENABLE_AUGMENTATION:
            image = cv2.imread(image_path)
            if image is None:
                continue
            # Randomly select 1 or 2 augmentations, including 'flip'
            num_augmented = random.randint(1, 2)
            chosen_augs = random.sample(list(augmentations.keys()) + ['flip'], num_augmented)
            for aug in chosen_augs:
                augmented_image_name = generate_unique_name(image_path, suffix=aug)
                augmented_image_path = os.path.join(output_structure[set_name]['images'], augmented_image_name)
                augmented_label_name = os.path.splitext(augmented_image_name)[0] + '.txt'
                augmented_label_path = os.path.join(output_structure[set_name]['labels'], augmented_label_name)

                if aug == 'flip':
                    # If the boolean is set, only flip if one of the specific classes is present.
                    if FLIP_ONLY_IF_SPECIFIC_CLASSES:
                        flip_flag = False
                        if os.path.exists(label_path):
                            with open(label_path, 'r') as f:
                                lines = f.readlines()
                            for line in lines:
                                parts = line.strip().split()
                                if len(parts) >= 1 and parts[0] in SPECIFIC_CLASSES:
                                    flip_flag = True
                                    break
                        if not flip_flag:
                            continue  # Skip flip augmentation if none of the specified classes are found.
                    flip_image_and_update_labels(image, label_path, augmented_image_path, augmented_label_path)
                else:
                    augmented_image = augmentations[aug](image)
                    cv2.imwrite(augmented_image_path, augmented_image)
                    # For other augmentations, copy label if it exists; otherwise, create an empty file.
                    if os.path.exists(label_path):
                        shutil.copy(label_path, augmented_label_path)
                    else:
                        open(augmented_label_path, 'w').close()


# Copy and augment for training and validation sets
copy_and_augment_files(train_images, train_labels, 'train')
copy_and_augment_files(val_images, val_labels, 'val')

print("Dataset successfully split into train and val sets with a special folder contribution of {:.1%}.".format(
    special_percentage))
