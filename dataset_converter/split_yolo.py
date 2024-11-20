import os
import random
import shutil

# Set paths
images_folder = '/home/simon/Documents/Master-Thesis/data/yolo_label_studio/images'
yolo_folder = ('/home/simon/Documents/Master-Thesis/data/yolo_training_data')
labels_folder = os.path.join(images_folder, '../labels')
output_folders = {
    'train': {'images': os.path.join(yolo_folder, 'images/train'), 'labels': os.path.join(yolo_folder, 'labels/train')},
    'val': {'images': os.path.join(yolo_folder, 'images/val'), 'labels': os.path.join(yolo_folder, 'labels/val')},
    'test': {'images': os.path.join(yolo_folder, 'images/test'), 'labels': os.path.join(yolo_folder, 'labels/test')}
}

# Create output directories
for set_name, paths in output_folders.items():
    os.makedirs(paths['images'], exist_ok=True)
    os.makedirs(paths['labels'], exist_ok=True)

# List all images in the folder
all_images = [f for f in os.listdir(images_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]
random.shuffle(all_images)

# Calculate dataset split indices
num_images = len(all_images)
train_size = int(num_images * 0.7)
val_size = int(num_images * 0.2)

train_images = all_images[:train_size]
val_images = all_images[train_size:train_size + val_size]
test_images = all_images[train_size + val_size:]


# Helper function to move files
def move_files(image_files, set_name):
    for image in image_files:
        image_path = os.path.join(images_folder, image)
        label_path = os.path.join(labels_folder, image.replace(os.path.splitext(image)[1], '.txt'))

        # Move image
        shutil.copy(image_path, output_folders[set_name]['images'])

        # Move corresponding label if it exists
        if os.path.exists(label_path):
            shutil.copy(label_path, output_folders[set_name]['labels'])


# Move files to respective folders
move_files(train_images, 'train')
move_files(val_images, 'val')
move_files(test_images, 'test')

print("Dataset successfully split into train, val, and test sets.")
