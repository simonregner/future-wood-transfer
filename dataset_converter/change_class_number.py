import os

# -------- CONFIGURATION --------
base_folder = "/home/simon/Documents/Master-Thesis/data/yolo_training_data/"
numbers_to_change = ["7"]# ["2", "3", "5", "6"]
special_number = "4"

classes_to_remove = []  # <- example: these class lines will be removed completely
# -------------------------------

changed_files_count = 0
removed_lines_count = 0

for root, _, files in os.walk(base_folder):
    for file in files:
        if file.endswith(".txt"):
            file_path = os.path.join(root, file)
            file_changed = False
            removed_lines = 0

            with open(file_path, "r") as f:
                lines = f.readlines()

            modified_lines = []
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                if parts[0] in classes_to_remove:
                    file_changed = True
                    removed_lines += 1
                    continue  # skip this line entirely
                if parts[0] in numbers_to_change:
                    parts[0] = special_number
                    file_changed = True
                modified_lines.append(" ".join(parts))

            if file_changed:
                with open(file_path, "w") as f:
                    f.write("\n".join(modified_lines) + ("\n" if modified_lines else ""))
                changed_files_count += 1
                removed_lines_count += removed_lines

print(f"Classes {', '.join(numbers_to_change)} have been changed to class {special_number} in {changed_files_count} file(s).")
print(f"{removed_lines_count} line(s) with classes {', '.join(classes_to_remove)} have been removed.")
