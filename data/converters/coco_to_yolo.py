#!/usr/bin/env python3
"""
Script to convert COCO annotations into YOLOv8 oriented bounding box (OBB) format.
Each output text file corresponds to an image and contains one line per annotation:
    <class_id> <x_center> <y_center> <width> <height> <angle>
Coordinates (x_center, y_center, width, height) are normalized by the image dimensions.
Angle is output in degrees.
"""

import os
import json
import numpy as np
import cv2

# Set your COCO JSON path and the output directory here
COCO_JSON_PATH = "/home/simon/Documents/Master-Thesis/data/COCO/annotations/panoptic_train2017.json"
OUTPUT_DIR = "/home/simon/Documents/Master-Thesis/data/COCO/annotation_yolo/train_2017"


def convert_segment(segment, image_info):
    """
    Convert a single panoptic segment to YOLOv8 OBB format.
    Uses the provided COCO bbox from the segment info.

    Returns:
        class_id (int), normalized cx, cy, w, h, and angle (float)
    """
    img_w = image_info['width']
    img_h = image_info['height']

    # Use the bbox from the segment info (expected format: [x, y, w, h])
    x, y, w, h = segment['bbox']
    cx = x + w / 2
    cy = y + h / 2

    # Normalize coordinates relative to image dimensions
    cx_norm = cx / img_w
    cy_norm = cy / img_h
    w_norm = w / img_w
    h_norm = h / img_h

    # Adjust class_id: subtract one to convert to 0-indexed if needed
    class_id = segment['category_id'] - 1

    # Since we don't have rotated segmentation, angle is set to 0
    angle = 0.0

    return class_id, cx_norm, cy_norm, w_norm, h_norm, angle


def convert_panoptic_to_yolov8(panoptic_json_path, output_dir):
    """
    Converts a COCO panoptic JSON file to YOLOv8 OBB format.
    For each image in the panoptic annotations, writes a .txt file with one line per segment.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load panoptic annotations
    with open(panoptic_json_path, 'r') as f:
        data = json.load(f)

    # Build a dictionary of images based on image id
    images_dict = {img['id']: img for img in data['images']}

    # The panoptic 'annotations' field is a list where each item corresponds to an image
    for ann in data['annotations']:
        image_id = ann['image_id']
        if image_id not in images_dict:
            continue
        image_info = images_dict[image_id]

        # Use the image file name (without extension) to name the output annotation file
        image_filename = image_info['file_name']
        txt_filename = os.path.splitext(image_filename)[0] + '.txt'
        output_path = os.path.join(output_dir, txt_filename)

        ann_lines = []
        # Process each segment in the image's segments_info list
        for segment in ann['segments_info']:
            class_id, cx, cy, w, h, angle = convert_segment(segment, image_info)
            line = f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {angle:.6f}"
            ann_lines.append(line)

        # Write annotation file only if there are segments for this image
        if ann_lines:
            with open(output_path, 'w') as f:
                f.write("\n".join(ann_lines))
    print(f"Converted annotations have been saved to: {output_dir}")


def main():
    convert_panoptic_to_yolov8(COCO_JSON_PATH, OUTPUT_DIR)


if __name__ == "__main__":
    main()
