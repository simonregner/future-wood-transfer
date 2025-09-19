import os

def delete_files_with_endings(root_folder, endings):
    """
    Delete files in subfolders if their filename ends with one of the endings before the extension.
    
    Args:
        root_folder (str): Path to the root folder.
        endings (list[str]): List of endings to match (e.g. ["_rain", "_snow"]).
    """
    deleted_files = []

    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            name, ext = os.path.splitext(filename)
            for ending in endings:
                if name.endswith(ending):
                    file_path = os.path.join(dirpath, filename)
                    try:
                        os.remove(file_path)
                        deleted_files.append(file_path)
                        print(f"Deleted: {file_path}")
                    except Exception as e:
                        print(f"Could not delete {file_path}: {e}")
                    break  # No need to check other endings

    print(f"\nTotal deleted files: {len(deleted_files)}")


# ====================2
# Example usage:
# ====================
if __name__ == "__main__":
    folder = "/home/simon/Documents/Master-Thesis/data/yolo_training_data_road_noaug"   # <-- change this to your folder
    endings_to_remove = ["_darker", "_brighter", "_noise", "_jitter", "_spring", "_fall", "_rain", "_snowflakes", "_fog"]        # endings before extension
    delete_files_with_endings(folder, endings_to_remove)
