import os

# Directory containing the text files for each image
directory_path = "/home/simon/Documents/Master-Thesis/data/yolo_label_studio/Road_Detection_Asphalt/labels"

# Iterate over all files in the directory
for filename in os.listdir(directory_path):
    if filename.endswith(".txt"):
        file_path = os.path.join(directory_path, filename)

        # Read the file lines
        with open(file_path, "r") as file:
            lines = file.readlines()

        # Modify lines where class is 1
        modified_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) > 0 and parts[0] == "0":
                parts[0] = "5"
            modified_lines.append(" ".join(parts))

        # Write the modified lines back to the file
        with open(file_path, "w") as file:
            file.write("\n".join(modified_lines))

print("Class 1 has been changed to class 5 for all text files in the directory.")