import os
import cv2
import shutil
import random
import hashlib
import numpy as np
from tqdm import tqdm

# -------------------- CONFIG -------------------- #
ROOT = "/home/simon/Documents/Master-Thesis/data/yolo_lanes"
OUTPUT = "/home/simon/Documents/Master-Thesis/data/yolo_training_data"
SPECIAL_IMAGES = "/home/simon/Documents/Master-Thesis/data/COCO/train2017"
SPECIAL_LABELS = "/home/simon/Documents/Master-Thesis/data/COCO/annotations/empty"
SPECIAL_PERC = 0.1
CLASS_MAP = {"2": "4", "3": "4", "5": "4", "6": "4"}
FLIP_CLASSES = {"2", "3", "5"}
AUGMENT = True
FLIP_IF_CLASS_MATCH = True
# ------------------------------------------------- #

# 🧹 Clean + Smooth
def clean_and_smooth(folder):
    for root, _, files in os.walk(folder):
        for file in files:
            if not file.endswith(".txt") or file == "classes.txt":
                continue
            txt_path = os.path.join(root, file)
            img_base = os.path.splitext(file)[0]
            img_dir = os.path.normpath(os.path.join(root, "../images"))
            img_path = None
            for ext in [".jpg", ".jpeg", ".png"]:
                p = os.path.join(img_dir, img_base + ext)
                if os.path.exists(p):
                    img_path = p
                    break

            if not img_path:
                os.remove(txt_path)
                continue

            with open(txt_path) as f:
                lines = f.readlines()
            keep = any(line.strip().startswith("7 ") for line in lines)
            if not keep:
                os.remove(txt_path)
                if os.path.exists(img_path):
                    os.remove(img_path)
                continue

            img = cv2.imread(img_path)
            h, w = img.shape[:2]
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                cls = parts[0]
                coords = list(map(float, parts[1:]))

                cls = CLASS_MAP.get(cls, cls)

                if cls != "7":
                    new_lines.append(" ".join([cls] + [f"{c:.6f}" for c in coords]))
                    continue

                pts = [(coords[i]*w, coords[i+1]*h) for i in range(0, len(coords), 2)]
                pts = np.array(pts, dtype=np.float32).reshape((-1, 1, 2))
                smoothed = cv2.approxPolyDP(pts, 0.001 * cv2.arcLength(pts, True), True)
                mask = np.zeros((h, w), dtype=np.uint8)
                if smoothed is None or len(smoothed) < 3:
                    continue  # Skip if polygon too small or invalid

                smoothed_int = smoothed.astype(np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(mask, [smoothed_int], 255)
                dilated = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
                contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    continue
                biggest = max(contours, key=cv2.contourArea)[:, 0, :]
                norm = [(x/w, y/h) for x, y in biggest]
                flat = " ".join(f"{x:.6f} {y:.6f}" for x, y in norm)
                new_lines.append(f"{cls} {flat}")
            with open(txt_path, "w") as f:
                f.write("\n".join(new_lines))

# 📦 Load data
def load_image_label_pairs(folder):
    images, labels = [], []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                img = os.path.join(root, f)
                lbl = os.path.join(root, "../labels", os.path.splitext(f)[0] + ".txt")
                images.append(img)
                labels.append(lbl)
    return images, labels

# 🧠 Augmentations
def flip_image(img, label_path, out_img_path, out_lbl_path):
    flipped = cv2.flip(img, 1)
    cv2.imwrite(out_img_path, flipped)
    if not os.path.exists(label_path):
        open(out_lbl_path, 'w').close()
        return
    with open(label_path) as f:
        lines = f.readlines()
    new = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 3 or len(parts[1:]) % 2 != 0:
            continue
        cls = parts[0]
        cls = "3" if cls == "2" else "2" if cls == "3" else cls
        coords = list(map(float, parts[1:]))
        flipped_coords = []
        for i in range(0, len(coords), 2):
            flipped_coords += [1.0 - coords[i], coords[i+1]]
        new.append(f"{cls} " + " ".join(f"{x:.6f}" for x in flipped_coords))
    with open(out_lbl_path, "w") as f:
        f.write("\n".join(new))

def generate_name(path, suffix=""):
    name = os.path.basename(path)
    folder = os.path.basename(os.path.dirname(path))
    h = hashlib.md5((folder + name).encode()).hexdigest()[:6]
    base, ext = os.path.splitext(name)
    return f"{base}_{h}_{suffix}{ext}" if suffix else f"{base}_{h}{ext}"

# 🪄 Copy + augment
def export_dataset(imgs, lbls, out_img, out_lbl):
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_lbl, exist_ok=True)
    for img, lbl in tqdm(zip(imgs, lbls), total=len(imgs), desc="Exporting"):
        img_out = os.path.join(out_img, generate_name(img))
        lbl_out = os.path.join(out_lbl, os.path.splitext(os.path.basename(img_out))[0] + ".txt")

        image = cv2.imread(img)
        if image is None:
            continue
        cv2.imwrite(img_out, image)
        if os.path.exists(lbl):
            shutil.copy(lbl, lbl_out)
        else:
            open(lbl_out, 'w').close()

        if AUGMENT:
            aug_name = generate_name(img, "flip")
            img_aug = os.path.join(out_img, aug_name)
            lbl_aug = os.path.join(out_lbl, os.path.splitext(aug_name)[0] + ".txt")

            if FLIP_IF_CLASS_MATCH:
                with open(lbl, 'r') as f:
                    if not any(line.split()[0] in FLIP_CLASSES for line in f if line.strip()):
                        continue

            flip_image(image, lbl, img_aug, lbl_aug)

# 🚀 Main Pipeline
def main():
    clean_and_smooth(ROOT)

    main_imgs, main_lbls = load_image_label_pairs(ROOT)
    sp_imgs, sp_lbls = load_image_label_pairs(SPECIAL_IMAGES)
    N = len(main_imgs)
    sp_N = min(len(sp_imgs), int((SPECIAL_PERC * N) / (1 - SPECIAL_PERC)))

    print(f"Main: {N}, Special: {sp_N}")
    sp_sel = random.sample(list(zip(sp_imgs, sp_lbls)), sp_N)
    imgs = main_imgs + [i for i, _ in sp_sel]
    lbls = main_lbls + [l for _, l in sp_sel]

    combined = list(zip(imgs, lbls))
    random.shuffle(combined)
    imgs, lbls = zip(*combined)

    train_len = int(0.7 * len(imgs))
    train_imgs, val_imgs = imgs[:train_len], imgs[train_len:]
    train_lbls, val_lbls = lbls[:train_len], lbls[train_len:]

    export_dataset(train_imgs, train_lbls, os.path.join(OUTPUT, "train/images"), os.path.join(OUTPUT, "train/labels"))
    export_dataset(val_imgs, val_lbls, os.path.join(OUTPUT, "val/images"), os.path.join(OUTPUT, "val/labels"))
    print("✅ Export complete.")

if __name__ == "__main__":
    main()
