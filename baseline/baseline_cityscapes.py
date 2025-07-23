import requests
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

from scipy.ndimage import label

import cv2


# Cityscapes label map (label_id to name)
CITYSCAPES_LABELS = {
    0: 'road', 1: 'ego vehicle', 2: 'rectification border', 3: 'out of roi', 4: 'static',
    5: 'dynamic', 6: 'ground', 7: 'road1', 8: 'sidewalk', 9: 'parking', 10: 'rail track',
    11: 'building', 12: 'wall', 13: 'fence', 14: 'guard rail', 15: 'bridge', 16: 'tunnel',
    17: 'pole', 18: 'polegroup', 19: 'traffic light', 20: 'traffic sign', 21: 'vegetation',
    22: 'terrain', 23: 'sky', 24: 'person', 25: 'rider', 26: 'car', 27: 'truck',
    28: 'bus', 29: 'caravan', 30: 'trailer', 31: 'train', 32: 'motorcycle', 33: 'bicycle', -1: 'void'
}

# Load processor and model
processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-cityscapes-panoptic")
model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-cityscapes-panoptic")

# Load a local image
image = Image.open("baseline/image_09.jpeg").convert("RGB")
inputs = processor(images=image, return_tensors="pt")

# Run inference
with torch.no_grad():
    outputs = model(**inputs)

# Post-process panoptic segmentation
result = processor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]

# Extract segmentation and segment info
segmentation_map = result["segmentation"].numpy()
segments_info = result["segments_info"]

road_ids = [
    segment["id"]
    for segment in segments_info
    if CITYSCAPES_LABELS.get(segment["label_id"], "") == "road"
]

road_mask = np.isin(segmentation_map, road_ids).astype(np.uint8)


# Assume road_mask is your binary mask (values 0 or 1)
labeled_mask, num_features = label(road_mask)

# Create individual binary masks
min_area = 1000

individual_masks = []
for i in range(1, num_features + 1):
    component = (labeled_mask == i).astype(np.uint8)
    area = component.sum()
    if area >= min_area:
        individual_masks.append(component)

def create_offset_masks_from_road(road_mask, step=20, radius=10):
    height, width = road_mask.shape
    print(height, width)
    left_points = []
    right_points = []

    # Step 1: scan from bottom to top every `step` rows
    for y in range(height - 1, 0, -step):
        x_indices = np.where(road_mask[y] > 0)[0]
        if len(x_indices) >= 2:
            left_x = x_indices[0]
            right_x = x_indices[-1]
            left_points.append((left_x, y))
            right_points.append((right_x, y))

    # Convert to numpy for drawing
    left_points_np = np.array(left_points, dtype=np.int32)
    right_points_np = np.array(right_points, dtype=np.int32)        

    # Step 2: Create empty masks
    left_mask = np.zeros_like(road_mask, dtype=np.uint8)
    right_mask = np.zeros_like(road_mask, dtype=np.uint8)

    # Step 3: Draw polylines
    if len(left_points_np) >= 2:
        cv2.polylines(left_mask, [left_points_np], isClosed=False, color=1, thickness=1)
        left_mask = cv2.dilate(left_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius*2, radius*2)))

    if len(right_points_np) >= 2:
        cv2.polylines(right_mask, [right_points_np], isClosed=False, color=1, thickness=1)
        right_mask = cv2.dilate(right_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius*2, radius*2)))

    return left_mask, right_mask, left_points, right_points

print(f"Number of individual road masks: {len(individual_masks)}")

outer_mask = []

for i, mask in enumerate(individual_masks):
    # Create offset masks from the road mask
    print(f"Processing individual mask {i+1}/{len(individual_masks)}")
    left_mask, right_mask, left_points, right_points = create_offset_masks_from_road(mask)

    # Add the combined mask to the list of individual masks
    outer_mask.append(left_mask)
    outer_mask.append(right_mask)

individual_masks.extend(outer_mask)


# Generate a unique color for each segment
id2color = [
    np.random.randint(0, 255, size=3) for i in individual_masks
]

# Create a blank RGB image to hold color-coded segmentation
colored_segmentation = np.zeros((*segmentation_map.shape, 3), dtype=np.uint8)

for i, mask in enumerate(individual_masks):
    color = id2color[i]
    colored_segmentation[mask == 1] = color


#for segment in segments_info:
#    mask = segmentation_map == segment["id"]
#    colored_segmentation[mask] = id2color[segment["id"]]

# Create legend patches with label names

# Plot
# fig, axs = plt.subplots(1, 2, figsize=(16, 8))
# axs[0].imshow(image)
# axs[0].set_title("Original Image")
# axs[0].axis("off")

# axs[1].imshow(colored_segmentation)
# axs[1].set_title("Panoptic Segmentation")
# axs[1].axis("off")

# plt.tight_layout()
# plt.show()



import os

# List of image paths
image_paths = [
    "baseline/image_01.png",
    "baseline/image_03.jpg",
    "baseline/image_05.png",
    "baseline/image_06.png",
    "baseline/image_07.png",
    "baseline/image_08.png",    
    "baseline/image_09.jpeg",
    "baseline/image_10.png",
    "baseline/image_11.png",
    "baseline/image_12.png"
]

rows = len(image_paths)
results = []  # To store image and segmentation pairs

for image_path in image_paths:
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    result = processor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
    segmentation_map = result["segmentation"].numpy()
    segments_info = result["segments_info"]

    road_ids = [
        segment["id"]
        for segment in segments_info
        if CITYSCAPES_LABELS.get(segment["label_id"], "") == "road"
    ]

    road_mask = np.isin(segmentation_map, road_ids).astype(np.uint8)

    labeled_mask, num_features = label(road_mask)

    min_area = 1000
    individual_masks = []
    for i in range(1, num_features + 1):
        component = (labeled_mask == i).astype(np.uint8)
        if component.sum() >= min_area:
            individual_masks.append(component)

    outer_mask = []
    for mask in individual_masks:
        left_mask, right_mask, _, _ = create_offset_masks_from_road(mask)
        outer_mask.extend([left_mask, right_mask])

    individual_masks.extend(outer_mask)

    id2color = [np.random.randint(0, 255, size=3) for _ in individual_masks]
    colored_segmentation = np.zeros((*segmentation_map.shape, 3), dtype=np.uint8)
    for i, mask in enumerate(individual_masks):
        colored_segmentation[mask == 1] = id2color[i]

    results.append((image, colored_segmentation))


# Plot all images and masks vertically
fig, axs = plt.subplots(rows, 2, figsize=(12, rows * 5))

if rows == 1:
    axs = np.expand_dims(axs, 0)  # Ensure axs is 2D for consistent access

for i, (orig_img, seg_img) in enumerate(results):
    axs[i, 0].imshow(orig_img)
    #axs[i, 0].set_title(f"Original Image {i+1}")
    axs[i, 0].axis("off")

    axs[i, 1].imshow(seg_img)
    #axs[i, 1].set_title(f"Segmented Mask {i+1}")
    axs[i, 1].axis("off")

plt.tight_layout()
plt.show()