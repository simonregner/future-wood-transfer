import os

# List of directory paths containing the text files for each image
directory_paths = [
    "/home/simon/Documents/Master-Thesis/data/yolo_training_data/train/labels/",
    "/home/simon/Documents/Master-Thesis/data/yolo_training_data/val/labels/"
    # Add more paths as needed
]
# 0: background
# 1: bulge_road
# 2: left_turn
# 3: right_turn
# 4: road
# 5: straight_turn
# List of class numbers to change and the special number to replace them with.
numbers_to_change = ["2", "3", "5", "6"]  # Change these to the numbers you want to replace
#numbers_to_change = ["4"]  # Change these to the numbers you want to replace
special_number = "4"                      # Change this to your desired special number

changed_files_count = 0  # Counter for the number of files that have been modified

# Process each directory in the list
for directory_path in directory_paths:
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory_path, filename)
            file_changed = False  # Flag to track if the current file has been changed

            # Read the file lines
            with open(file_path, "r") as file:
                lines = file.readlines()

            # Modify lines where the first element (class number) is in the list
            modified_lines = []
            for line in lines:
                parts = line.strip().split()
                # Check if the line is non-empty and its first element is one of the numbers to change
                if parts and parts[0] in numbers_to_change:
                    parts[0] = special_number
                    file_changed = True
                modified_lines.append(" ".join(parts))

            # Write the modified lines back to the file
            with open(file_path, "w") as file:
                file.write("\n".join(modified_lines))

            if file_changed:
                changed_files_count += 1

print(f"Classes {', '.join(numbers_to_change)} have been changed to class {special_number} in {changed_files_count} file(s) across the specified directories.")
