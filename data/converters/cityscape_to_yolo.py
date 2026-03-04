import os
import json
from glob import glob
from pathlib import Path
from tqdm import tqdm

# Debug: print current directory
print("Current working directory:", os.getcwd())

# ----------- CONFIGURATION -----------
input_root = "../data/cityscape/original/gtCoarse/train_extra/"
output_root = "../data/cityscape/labels/"
# You define the classes and their desired YOLO class IDs here
class_to_id = {
    "road": 4,
}
selected_classes = list(class_to_id.keys())
# -------------------------------------

def convert_polygon_to_yolo(polygon, img_width, img_height):
    return [(x / img_width, y / img_height) for x, y in polygon]

def save_yolo_txt(yolo_data, output_path):
    with open(output_path, "w") as f:
        for class_id, polygon in yolo_data:
            flat_coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in polygon)
            f.write(f"{class_id} {flat_coords}\n")

def process_json_file(json_path):
    with open(json_path) as f:
        data = json.load(f)

    img_width = data["imgWidth"]
    img_height = data["imgHeight"]
    objects = data["objects"]

    yolo_annotations = []

    for obj in objects:
        label = obj["label"]
        if label in selected_classes:
            yolo_poly = convert_polygon_to_yolo(obj["polygon"], img_width, img_height)
            yolo_annotations.append((class_to_id[label], yolo_poly))

    if yolo_annotations:
        base_name = Path(json_path).stem.replace("_gtCoarse_polygons", "")
        output_path = Path(output_root) / f"{base_name}.txt"
        save_yolo_txt(yolo_annotations, output_path)

def main():
    os.makedirs(output_root, exist_ok=True)

    # DEBUG: Check glob result
    test_glob = glob(f"{input_root}/**/*.json", recursive=True)
    print(f"Found {len(test_glob)} JSON files from: {input_root}")

    for json_file in tqdm(test_glob, desc="Converting"):
        process_json_file(json_file)

if __name__ == "__main__":
    main()
