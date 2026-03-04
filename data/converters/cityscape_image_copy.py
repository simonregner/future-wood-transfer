import os
import shutil
from glob import glob
from pathlib import Path
from PIL import Image
from tqdm import tqdm  # ← Loading bar

# ----------- CONFIGURATION -----------
input_root = "../../../../../media/simon/T7 Shield/leftImg8bit_trainextra(1)/"
output_folder = "../data/cityscape/images/"
remove_text = "_leftImg8bit"
resize_factor = 2 / 3
# --------------------------------------

def main():
    os.makedirs(output_folder, exist_ok=True)

    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob(f"{input_root}/**/{ext}", recursive=True))

    print(f"Found {len(image_files)} image files.")

    for img_path in tqdm(image_files, desc="Processing images"):
        original_name = Path(img_path).name
        new_name = original_name.replace(remove_text, "")
        new_path = Path(output_folder) / new_name

        # Skip if file already exists
        if new_path.exists():
            continue

        try:
            with Image.open(img_path) as img:
                new_size = (
                    int(img.width * resize_factor),
                    int(img.height * resize_factor)
                )
                resized_img = img.resize(new_size, Image.LANCZOS)
                resized_img.save(new_path)
        except Exception as e:
            print(f"Failed to process {img_path}: {e}")

    print("Done resizing, renaming, and copying.")

if __name__ == "__main__":
    main()
