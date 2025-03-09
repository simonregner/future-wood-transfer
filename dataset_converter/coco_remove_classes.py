import os

# === Configuration ===
# Folder containing your images
IMAGE_DIR = "/home/simon/Documents/Master-Thesis/data/COCO/train2017"
# Folder containing YOLOv8 OBB annotation files (.txt)
ANN_DIR = "/home/simon/Documents/Master-Thesis/data/COCO/annotation_yolo/train_2017"
# Folder where empty annotation files for the remaining images will be created
NEW_ANN_DIR = "/home/simon/Documents/Master-Thesis/data/COCO/annotation_yolo/empty"
# List of classes (as strings) that should trigger the removal
CLASSES_TO_REMOVE = ['2', '3', '4', '6', '8', '10', '11', '12', '13', '14']  # Change these to the classes you want to remove
# If True, the script will also remove the annotation file
REMOVE_ANNOTATION = False

def remove_files_with_classes(image_dir, ann_dir, classes_to_remove, remove_annotation=True):
    """
    For each annotation file, if any line starts with a class in classes_to_remove,
    remove the corresponding image (and optionally the annotation file).
    """
    ann_files = [f for f in os.listdir(ann_dir) if f.endswith('.txt')]
    removed_count = 0

    for ann_file in ann_files:
        ann_path = os.path.join(ann_dir, ann_file)
        try:
            with open(ann_path, 'r') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"Error reading {ann_path}: {e}")
            continue

        remove = False
        for line in lines:
            tokens = line.strip().split()
            if not tokens:
                continue
            obj_class = tokens[0]
            if obj_class in classes_to_remove:
                remove = True
                break

        if remove:
            # Assume image file has the same basename and one of common image extensions
            basename = os.path.splitext(ann_file)[0]
            found_image = False
            for ext in ['.jpg', '.jpeg', '.png']:
                image_path = os.path.join(image_dir, basename + ext)
                if os.path.exists(image_path):
                    try:
                        os.remove(image_path)
                        print(f"Removed image: {image_path}")
                        found_image = True
                        removed_count += 1
                    except Exception as e:
                        print(f"Error removing {image_path}: {e}")
                    break
            if not found_image:
                print(f"No matching image found for annotation {ann_file}")

            if remove_annotation:
                try:
                    os.remove(ann_path)
                    print(f"Removed annotation: {ann_path}")
                except Exception as e:
                    print(f"Error removing {ann_path}: {e}")

    print(f"Total images removed: {removed_count}")

def create_empty_annotation_files(image_dir, new_ann_dir):
    """
    For every image in image_dir, create an empty annotation file in new_ann_dir.
    """
    # Create the new annotation directory if it does not exist
    if not os.path.exists(new_ann_dir):
        os.makedirs(new_ann_dir)
        print(f"Created directory {new_ann_dir}")

    # Consider common image extensions
    image_extensions = ['.jpg', '.jpeg', '.png']
    image_files = [f for f in os.listdir(image_dir) if any(f.endswith(ext) for ext in image_extensions)]
    created_count = 0

    for image_file in image_files:
        basename = os.path.splitext(image_file)[0]
        new_ann_path = os.path.join(new_ann_dir, basename + ".txt")
        try:
            # Create an empty file (overwrites if it already exists)
            with open(new_ann_path, 'w') as f:
                pass
            print(f"Created empty annotation file: {new_ann_path}")
            created_count += 1
        except Exception as e:
            print(f"Error creating file {new_ann_path}: {e}")

    print(f"Total empty annotation files created: {created_count}")

if __name__ == "__main__":
    # Step 1: Remove images (and annotation files, if flagged) with the specified classes.
    remove_files_with_classes(IMAGE_DIR, ANN_DIR, CLASSES_TO_REMOVE, REMOVE_ANNOTATION)
    # Step 2: For every remaining image, create an empty annotation file in the new folder.
    create_empty_annotation_files(IMAGE_DIR, NEW_ANN_DIR)