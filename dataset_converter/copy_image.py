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

def copy_and_rename_images(image_paths, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    current_number = get_highest_number_in_path(output_path)

    for path in image_paths:
        if os.path.exists(path):
            files = sorted(os.listdir(path))
            for file_name in files:
                file_path = os.path.join(path, file_name)
                if os.path.isfile(file_path):
                    ext = os.path.splitext(file_name)[1]
                    new_name = f"{current_number:06d}{ext}"
                    shutil.copy(file_path, os.path.join(output_path, new_name))
                    current_number += 1

# Example usage
image_paths = [
    '/home/simon/Documents/Master-Thesis/CaSSeD/CaSSed_Dataset_Final/real_world_data/Dataset1_Segmentation/Dataset1B - Powerline/raw_images',
    '/home/simon/Documents/Master-Thesis/CaSSeD/CaSSed_Dataset_Final/real_world_data/Dataset1A-Brown_field/raw_images',
    '/home/simon/Documents/Master-Thesis/CaSSeD/CaSSed_Dataset_Final/real_world_data/Dataset2_Fogdata_Segmentation/raw_images',
    '/home/simon/Documents/Master-Thesis/CaSSeD/CaSSed_Dataset_Final/real_world_data/Dataset3_NorthFarm_Segmentation/raw_images',
    '/home/simon/Documents/Master-Thesis/CaSSeD/CaSSed_Dataset_Final/real_world_data/Dataset4_NorthSlope_Segmentation/Dataset1/raw_images',
    '/home/simon/Documents/Master-Thesis/CaSSeD/CaSSed_Dataset_Final/real_world_data/Dataset4_NorthSlope_Segmentation/Dataset2/raw_images']

output_path = "/home/simon/Documents/Master-Thesis/data/all_images"
copy_and_rename_images(image_paths, output_path)
