import os
import cv2
import random
import numpy as np
from pathlib import Path

# Paths
DATASET_PATH = "/home/simon/Documents/Master-Thesis/data/yolo_training_data"
AUGMENTATION_PATH = "/home/simon/Documents/Master-Thesis/data/yolo_label_studio/data_augmentation"
IMAGE_FOLDER_NAME = 'images'
LABEL_FOLDER_NAME = 'labels'

# Brightness adjustment factors
BRIGHTNESS_DARK_FACTOR = 0.5  # Darker image factor
BRIGHTNESS_BRIGHT_FACTOR = 1.5  # Brighter image factor
NOISE_STD_DEV = 25  # Standard deviation for Gaussian noise


def create_directories(augmentation_path):
    """Create augmented images and labels directories in the main augmentations folder."""
    image_path = os.path.join(augmentation_path, 'images')
    label_path = os.path.join(augmentation_path, 'labels')
    os.makedirs(image_path, exist_ok=True)
    os.makedirs(label_path, exist_ok=True)


def adjust_brightness(image, factor):
    """Adjust the brightness of an image."""
    return cv2.convertScaleAbs(image, alpha=factor, beta=0)


def add_gaussian_noise(image, std_dev):
    """Add Gaussian noise to the image."""
    noise = np.random.normal(0, std_dev, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    return noisy_image


def jitter_colors(image):
    """Apply random brightness, contrast, and saturation changes to the image."""
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    random_brightness = random.uniform(0.7, 1.3)
    random_saturation = random.uniform(0.7, 1.3)
    hsv_image[:, :, 2] = np.clip(hsv_image[:, :, 2] * random_brightness, 0, 255)
    hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * random_saturation, 0, 255)
    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)


def augment_image(image_path, annotation_path, output_image_path, output_label_path, suffix, augmented_image):
    """Save the augmented image and copy the corresponding YOLO annotation file."""
    image_filename = f"{Path(image_path).stem}_{suffix}.jpg"
    label_filename = f"{Path(image_path).stem}_{suffix}.txt"

    # Paths to save images and labels
    image_output_path = os.path.join(output_image_path, image_filename)
    label_output_path = os.path.join(output_label_path, label_filename)

    # Save the augmented image
    cv2.imwrite(image_output_path, augmented_image)

    # Copy the corresponding YOLO annotation file
    if os.path.exists(annotation_path):
        with open(annotation_path, 'r') as file:
            annotation_content = file.read()

        with open(label_output_path, 'w') as file:
            file.write(annotation_content)


def apply_augmentations(image, augmentations):
    """Apply a series of augmentations to an image."""
    augmented_image = image.copy()
    for aug in augmentations:
        if aug == 'darker':
            augmented_image = adjust_brightness(augmented_image, BRIGHTNESS_DARK_FACTOR)
        elif aug == 'brighter':
            augmented_image = adjust_brightness(augmented_image, BRIGHTNESS_BRIGHT_FACTOR)
        elif aug == 'noise':
            augmented_image = add_gaussian_noise(augmented_image, NOISE_STD_DEV)
        elif aug == 'jitter':
            augmented_image = jitter_colors(augmented_image)
    return augmented_image


def augment_yolo_dataset(dataset_path, augmentation_path):
    """Apply multiple augmentations to each image in the YOLOv1 dataset."""
    image_extensions = ['.jpg', '.jpeg', '.png']
    augmentations = ['darker', 'brighter', 'noise', 'jitter']

    augmented_image_folder = os.path.join(augmentation_path, 'images')
    augmented_label_folder = os.path.join(augmentation_path, 'labels')

    for subfolder in ['train', 'val']:
        image_folder = os.path.join(dataset_path, subfolder, IMAGE_FOLDER_NAME)
        label_folder = os.path.join(dataset_path, subfolder, LABEL_FOLDER_NAME)

        for root, _, files in os.walk(image_folder):
            for file in files:
                if Path(file).suffix.lower() in image_extensions:
                    image_path = os.path.join(root, file)
                    annotation_path = os.path.join(label_folder, f"{Path(file).stem}.txt")

                    image = cv2.imread(image_path)
                    if image is None:
                        print(f"Failed to read image: {image_path}")
                        continue

                    # Select up to 2 augmentations randomly
                    selected_augmentations = random.sample(augmentations, random.randint(1, 2))
                    suffix = "_".join(selected_augmentations)

                    augmented_image = apply_augmentations(image, selected_augmentations)
                    augment_image(image_path, annotation_path, augmented_image_folder, augmented_label_folder, suffix,
                                  augmented_image)

                    print(f"Augmented: {image_path} with {selected_augmentations}")


if __name__ == "__main__":
    create_directories(AUGMENTATION_PATH)
    augment_yolo_dataset(DATASET_PATH, AUGMENTATION_PATH)