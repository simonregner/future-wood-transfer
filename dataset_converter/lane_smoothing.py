import os
import cv2
import numpy as np

def read_yolo_txt(txt_path):
    with open(txt_path, "r") as f:
        lines = f.readlines()

    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if line:
            parts = list(map(float, line.split()))
            if parts:
                cleaned_lines.append(parts)

    return cleaned_lines

def write_yolo_txt(txt_path, lines):
    with open(txt_path, "w") as f:
        for line in lines:
            class_id = int(line[0])
            coords = line[1:]
            coords_str = " ".join(f"{c:.6f}" for c in coords)
            f.write(f"{class_id} {coords_str}\n")

def smooth_polygon(points, epsilon_factor=0.01):
    contour = np.array(points, dtype=np.float32).reshape((-1, 1, 2))
    epsilon = epsilon_factor * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    return approx.reshape((-1, 2))

def get_image_dimensions(txt_path):
    txt_dir = os.path.dirname(txt_path)
    txt_file = os.path.splitext(os.path.basename(txt_path))[0]
    img_path_base = os.path.normpath(os.path.join(txt_dir, "../images", txt_file))

    for ext in [".jpg", ".jpeg", ".png"]:
        full_path = img_path_base + ext
        if os.path.exists(full_path):
            img = cv2.imread(full_path)
            if img is not None:
                return img.shape[1], img.shape[0]  # width, height
    print(f"⚠️ Image not found for {txt_path}")
    return None, None

def dilate_polygon(polygon, image_width, image_height, dilation_pixels=2):
    # Create empty mask
    mask = np.zeros((image_height, image_width), dtype=np.uint8)

    # Convert polygon to int32 for drawing
    polygon_int = np.array(polygon, dtype=np.int32)
    polygon_int = polygon_int.reshape((-1, 1, 2))

    # Draw filled polygon
    cv2.fillPoly(mask, [polygon_int], 255)

    # Dilate the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilation_pixels + 1, 2 * dilation_pixels + 1))
    dilated_mask = cv2.dilate(mask, kernel)

    # Find contours from the dilated mask
    contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Pick the largest contour by area
    if not contours:
        return polygon  # fallback
    largest_contour = max(contours, key=cv2.contourArea)
    return largest_contour[:, 0, :]  # Remove extra dimensions


def process_segmentation_file(txt_path):
    image_width, image_height = get_image_dimensions(txt_path)
    if image_width is None or image_height is None:
        return  # Skip if image not found

    lines = read_yolo_txt(txt_path)
    if not lines:
        print(f"⚠️ Skipping empty file: {txt_path}")
        return

    new_lines = []

    for line in lines:
        class_id = int(line[0])
        coords = line[1:]
        if class_id != 7:
            new_lines.append(line)
            continue

        polygon = [(coords[i]*image_width, coords[i+1]*image_height) for i in range(0, len(coords), 2)]
        smoothed = smooth_polygon(polygon, epsilon_factor=0.001)

        dilated = dilate_polygon(smoothed, image_width, image_height, dilation_pixels=2)

        resampled = smooth_polygon(dilated, epsilon_factor=0.001)

        resampled_norm = [(x / image_width, y / image_height) for x, y in resampled]
        flat = [coord for point in resampled_norm for coord in point]
        new_line = [class_id] + flat
        new_lines.append(new_line)

    write_yolo_txt(txt_path, new_lines)

def process_all_txt_files(root_folder):
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith(".txt") and filename != "classes.txt":
                txt_path = os.path.join(dirpath, filename)
                print(f"✅ Processing: {txt_path}")
                process_segmentation_file(txt_path)

# 🔧 Use it like this:
process_all_txt_files('/home/simon/Documents/Master-Thesis/data/yolo_lanes_smoothed')
