import os
import cv2
import numpy as np
from shapely.geometry import Polygon
from skimage import measure
import random


def mask_to_polygons(mask):
    """
    Convert a segmentation mask to polygons using contours.
    """
    contours = measure.find_contours(mask, 0.5, positive_orientation='low')
    polygons = []
    for contour in contours:
        poly = Polygon(contour)
        if poly.is_valid and poly.area > 10:  # Filter very small regions
            polygons.append(poly)
    return polygons


def convert_grey_to_white(mask):
    """
    Convert grey pixels to white and all other pixels to black.
    Args:
        mask: RGBA image as numpy array.
    Returns:
        A binary mask where grey pixels are white, and others are black.
    """
    binary_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
    grey_pixels = (mask[:, :, 0] == 154) & (mask[:, :, 1] == 154) & (mask[:, :, 2] == 155)
    binary_mask[grey_pixels] = 255
    return binary_mask


def create_coco8_annotation(dataset_paths, output_image_folders):
    """
    Converts segmentation masks to YOLOv8 COCO-seg format and splits data into train, val, and test.

    Args:
        dataset_paths: List of paths to datasets containing image and annotation folders.
        output_image_folders: List of folders to save images and corresponding txt files for train, val, and test datasets.
    """
    split_ratios = [0.7, 0.2, 0.1]
    split_indices = [0, 0, 0]

    for dataset_path in dataset_paths:
        # Paths for images and annotations
        images_path = os.path.join(dataset_path, "raw_images")
        annotations_path = os.path.join(dataset_path, "annotations")

        print(dataset_path)

        # Loop through each image in the dataset
        for img_file in os.listdir(images_path):
            if img_file.endswith(('.png', '.jpg', '.jpeg')):
                img = cv2.imread(os.path.join(images_path, img_file))
                height, width, _ = img.shape

                # Randomly assign to train, val, or test
                rand = random.random()
                if rand < split_ratios[0]:
                    split_idx = 0
                elif rand < split_ratios[0] + split_ratios[1]:
                    split_idx = 1
                else:
                    split_idx = 2

                img_id = split_indices[split_idx] + 1
                split_indices[split_idx] += 1

                # Define the new image path and copy the image
                new_img_name = f"{img_id}.jpg"
                new_img_path = os.path.join(output_image_folders[split_idx], new_img_name)
                os.makedirs(output_image_folders[split_idx], exist_ok=True)
                cv2.imwrite(new_img_path, img)

                # Corresponding annotation file
                annotation_file = os.path.join(annotations_path, img_file)
                if os.path.exists(annotation_file):
                    annotation_img = cv2.imread(annotation_file, cv2.IMREAD_UNCHANGED)  # Read RGBA image

                    # Extract road class (assuming road is represented by a specific class ID, e.g., 3)
                    road_category_id = 2

                    target_color = np.array([150, 150, 150])
                    target_color_upper = np.array([160, 160, 160])

                    road_mask = cv2.inRange(annotation_img, target_color, target_color_upper)
                    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE,
                                                 np.ones((5, 5), np.uint8))  # Fill holes in the mask

                    road_polygons = mask_to_polygons(road_mask)
                    txt_filename = os.path.splitext(new_img_name)[0] + '.txt'
                    txt_folder = os.path.join('/home/simon/Documents/Master-Thesis/data/coco8/labels',
                                              ['train', 'val', 'test'][split_idx])
                    os.makedirs(txt_folder,
                                exist_ok=True)  # Create specific annotations directory (train/val/test) if not exists
                    txt_file_path = os.path.join(txt_folder, txt_filename)

                    with open(txt_file_path, 'w') as txt_file:
                        for poly in road_polygons:
                            segmentation = np.array(poly.exterior.coords).ravel().tolist()
                            segmentation = [max(0, min(segmentation[i] / width, 1)) if i % 2 == 0 else max(0,
                                                                                                           min(
                                                                                                               segmentation[
                                                                                                                   i] / height,
                                                                                                               1)) for i
                                            in range(len(segmentation))]

                            # Write the category_id and segmentation information in YOLOv8 format
                            txt_file.write(f"{road_category_id} {' '.join(map(str, segmentation))}\n")



# Example usage:
dataset_paths = [
    '/home/simon/Documents/Master-Thesis/CaSSeD/CaSSed_Dataset_Final/real_world_data/Dataset1_Segmentation/Dataset1B - Powerline',
    '/home/simon/Documents/Master-Thesis/CaSSeD/CaSSed_Dataset_Final/real_world_data/Dataset1A-Brown_field',
    '/home/simon/Documents/Master-Thesis/CaSSeD/CaSSed_Dataset_Final/real_world_data/Dataset2_Fogdata_Segmentation',
    '/home/simon/Documents/Master-Thesis/CaSSeD/CaSSed_Dataset_Final/real_world_data/Dataset3_NorthFarm_Segmentation',
    '/home/simon/Documents/Master-Thesis/CaSSeD/CaSSed_Dataset_Final/real_world_data/Dataset4_NorthSlope_Segmentation/Dataset1',
    '/home/simon/Documents/Master-Thesis/CaSSeD/CaSSed_Dataset_Final/real_world_data/Dataset4_NorthSlope_Segmentation/Dataset2'
]
output_image_folders = [
    '/home/simon/Documents/Master-Thesis/data/coco8/images/train',
    '/home/simon/Documents/Master-Thesis/data/coco8/images/val',
    '/home/simon/Documents/Master-Thesis/data/coco8/images/test'
]

create_coco8_annotation(dataset_paths, output_image_folders)
