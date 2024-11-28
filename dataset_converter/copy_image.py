import os
import shutil

def get_highest_number_in_path(output_path):
    files = os.listdir(output_path)
    numbers = []
    for file in files:
        name, ext = os.path.splitext(file)
        if name.isdigit():
            numbers.append(int(name))
    return max(numbers) if numbers else 1

def copy_and_rename_images(image_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    current_number = get_highest_number_in_path(output_path)

    # Loop through all folders inside the image_path folder
    for root, dirs, files in os.walk(image_path):
        print(dirs)
        for file_name in sorted(files):
            # Check if the file ends with "vis.png"
            if file_name.endswith("vis.png"):
                file_path = os.path.join(root, file_name)
                if os.path.isfile(file_path):
                    ext = os.path.splitext(file_name)[1]
                    new_name = f"{current_number:06d}{ext}"
                    shutil.copy(file_path, os.path.join(output_path, new_name))
                    current_number += 1

# Example usage
image_path = '/home/simon/Documents/Master-Thesis/data/GOOSE/downloaded/train/'
output_path = "/home/simon/Documents/Master-Thesis/data/GOOSE/images"
copy_and_rename_images(image_path, output_path)