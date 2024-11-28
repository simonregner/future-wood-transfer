import os
import random
import shutil
import hashlib

# Set paths
input_folders = [
    '/home/simon/Documents/Master-Thesis/data/yolo_label_studio/CAVS/images',
    '/home/simon/Documents/Master-Thesis/data/yolo_label_studio/GOOSE/images',
]
output_folder = '/home/simon/Documents/Master-Thesis/data/yolo_training_data'

output_structure = {
    'train': {'images': os.path.join(output_folder, 'train/images'), 'labels': os.path.join(output_folder, 'train/labels')},
    'val': {'images': os.path.join(output_folder, 'val/images'), 'labels': os.path.join(output_folder, 'val/labels')}
}

# Create output directories
for set_name, paths in output_structure.items():
    os.makedirs(paths['images'], exist_ok=True)
    os.makedirs(paths['labels'], exist_ok=True)

# Collect all images and labels from input folders
all_images = []
all_labels = []

for folder in input_folders:
    images = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(('.jpg', '.png', '.jpeg'))]
    labels = [os.path.join(folder, '../labels', os.path.splitext(f)[0] + '.txt') for f in os.listdir(folder) if f.endswith(('.jpg', '.png', '.jpeg'))]
    all_images.extend(images)
    all_labels.extend(labels)

# Shuffle images and labels together (maintain order)
indices = list(range(len(all_images)))
random.shuffle(indices)
all_images = [all_images[i] for i in indices]
all_labels = [all_labels[i] for i in indices]

# Calculate dataset split indices
num_images = len(all_images)
train_size = int(num_images * 0.7)

train_images = all_images[:train_size]
val_images = all_images[train_size:]

train_labels = all_labels[:train_size]
val_labels = all_labels[train_size:]

# Helper function to generate unique file names
def generate_unique_name(file_path):
    file_name = os.path.basename(file_path)
    folder_name = os.path.basename(os.path.dirname(file_path))
    unique_hash = hashlib.md5((folder_name + file_name).encode()).hexdigest()[:8]
    new_file_name = f"{os.path.splitext(file_name)[0]}_{unique_hash}{os.path.splitext(file_name)[1]}"
    return new_file_name

# Helper function to copy files with unique names
def copy_files_with_unique_names(images, labels, set_name):
    for image_path, label_path in zip(images, labels):
        unique_image_name = generate_unique_name(image_path)
        unique_label_name = os.path.splitext(unique_image_name)[0] + '.txt'

        # Copy image
        output_image_path = os.path.join(output_structure[set_name]['images'], unique_image_name)
        shutil.copy(image_path, output_image_path)

        # Copy label if it exists
        if os.path.exists(label_path):
            output_label_path = os.path.join(output_structure[set_name]['labels'], unique_label_name)
            shutil.copy(label_path, output_label_path)

# Copy files to respective folders
copy_files_with_unique_names(train_images, train_labels, 'train')
copy_files_with_unique_names(val_images, val_labels, 'val')

print("Dataset successfully split into train and val sets with unique filenames.")