import os
from shapely.geometry import Polygon
from shapely.ops import unary_union
from tqdm import tqdm



def combine_classes_in_file(filepath):
    """
    Reads a YOLOv8 annotation file in place, extracts annotations for classes 4, 5, and 6,
    combines them (if any exist), and then writes back to the same file:
      - Keeping annotations from other classes.
      - Removing annotations for classes 4, 5, and 6.
      - Adding a new annotation (or annotations) with class 7 for the combined geometry.

    The expected annotation format per line is:
      class_id x1 y1 x2 y2 ... xN yN
    """
    classes_to_combine = {'2', '3', '4', '5'}
    kept_lines = []  # Lines to keep (annotations not in classes 4, 5, or 6)
    polygons_to_combine = []  # Polygons from classes 4, 5, and 6

    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            class_id = parts[0]
            if class_id in classes_to_combine:
                try:
                    coords = list(map(float, parts[1:]))
                    polygon_coords = list(zip(coords[0::2], coords[1::2]))
                    poly = Polygon(polygon_coords)
                    if not poly.is_valid:
                        poly = poly.buffer(0)
                    polygons_to_combine.append(poly)
                except Exception as e:
                    print(f"Error processing polygon in {filepath}: {parts[1:]} - {e}")
            else:
                kept_lines.append(line.strip())

    # If any polygons exist for classes 4, 5, 6, combine them
    if polygons_to_combine:
        combined = unary_union(polygons_to_combine)
        new_annotation_lines = []
        if combined.geom_type == 'Polygon':
            coords = list(combined.exterior.coords)
            coords_flat = " ".join(f"{coord:.6f}" for point in coords for coord in point)
            new_annotation_lines.append(f"6 {coords_flat}")
        elif combined.geom_type == 'MultiPolygon':
            # Use the .geoms attribute to iterate over each polygon in the MultiPolygon
            for poly in combined.geoms:
                coords = list(poly.exterior.coords)
                coords_flat = " ".join(f"{coord:.6f}" for point in coords for coord in point)
                new_annotation_lines.append(f"4 {coords_flat}")
        else:
            print(f"Combined geometry in {filepath} is not a polygon or multipolygon.")
            new_annotation_lines = []
        kept_lines.extend(new_annotation_lines)

    # Overwrite the file with the updated annotations
    with open(filepath, 'w') as f:
        for line in kept_lines:
            f.write(line + "\n")
    #print(f"Processed (in-place): {filepath}")


def process_folder_in_place(folder):
    """
    Processes all .txt annotation files in the given folder, modifying each file in place.
    """
    print(f"Processing folder: {folder}")
    for filename in tqdm(os.listdir(folder), total=len(os.listdir(folder)),desc=f"Create Intersection "): 
        if filename.lower().endswith(".txt"):
            filepath = os.path.join(folder, filename)
            combine_classes_in_file(filepath)


def process_two_folders_in_place(folder1, folder2):
    """
    Processes two separate folders, modifying the files in each folder in place.
    """
    process_folder_in_place(folder1)
    process_folder_in_place(folder2)


# Example usage with two folders:
folder1 = "/home/simon/Documents/Master-Thesis/data/yolo_training_data/train/labels"
folder2 = "/home/simon/Documents/Master-Thesis/data/yolo_training_data/val/labels"
#folder3 = "/home/simon/Documents/Master-Thesis/data/baseline_test/test/labels"
#process_folder_in_place(folder3)
# 
process_two_folders_in_place(folder1, folder2)